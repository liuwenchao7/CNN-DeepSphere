#ifndef WFSampling_cc
#define WFSampling_cc

// ---------------- C++ Standard Library ----------------
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

// ---------------- ROOT Library ----------------
#include <TTimeStamp.h>
#include <TSystem.h>
#include "TROOT.h"

// ---------------- Framework & Svc ----------------
#include "WFSampling.h"
#include "SniperKernel/AlgFactory.h"
#include "SniperKernel/SniperLog.h"
#include "SniperKernel/Task.h"
#include "SniperKernel/SniperPtr.h"
#include "EvtNavigator/EvtNavHelper.h"
#include "BufferMemMgr/IDataMemMgr.h"
#include "RootWriter/RootWriter.h"
#include "PMTCalibSvc/IPMTCalibSvc.hh"
#include "Geometry/IPMTParamSvc.h"

// ---------------- Event Data Model (EDM) ----------------
#include "Event/CdWaveformHeader.h"
#include "Event/CdLpmtCalibHeader.h"
#include "Identifier/CdID.h"
#include "Event/WpWaveformHeader.h"
#include "Event/WpCalibHeader.h"
#include "Identifier/WpID.h"
#include "EDMPath/EDMPath.hh"
#include "Event/CdTriggerHeader.h"

using namespace std;

DECLARE_ALGORITHM(WFSampling);

WFSampling::WFSampling(const std::string& name) 
  : AlgBase(name), 
    m_totalPMT(17612), 
    m_bl(100), 
    m_thr(15.0f), 
    m_thrh_h(50.0f), 
    m_thrw_h(15), 
    m_thrh_n(50.0f), 
    m_thrw_n(15), 
    m_rise_step(2),  
    m_polarity(-1),
    m_cnv_unit(1.0f),
    m_cnv_high_range(1.0f),
    m_thr_flat(15.0f),
    m_enableUserOutput(false),
    m_decon_mode(false), // <=== 新增：初始化 decon_mode
    m_CalcMeanGain(CalcMeanGain::FitSpectral),
    m_output_edm_path(""),
    m_c(1),
    m_memMgr(nullptr), 
    m_alg("stt"), 
    m_stat(0),
    m_process_cd(true)
{
    declProp("TotalPMT", m_totalPMT);
    declProp("BslBufLen", m_bl);
    declProp("Threshold", m_thr);
    declProp("Threshold_Hama_h", m_thrh_h);   
    declProp("Threshold_Hama_w", m_thrw_h);     
    declProp("Threshold_NNVT_h", m_thrh_n);   
    declProp("Threshold_NNVT_w", m_thrw_n); 
    declProp("RisingEdgeStep", m_rise_step);  
    declProp("Polarity", m_polarity);
    declProp("ChargeUnitConvertor", m_cnv_unit);
    declProp("HighRangeConvertor", m_cnv_high_range);
    declProp("ThresholdFlat", m_thr_flat);
    declProp("EnableUserOutput", m_enableUserOutput);
    declProp("DeconMode", m_decon_mode); // <=== 新增：暴露给 python 接口
    declProp("CalcMeanGain", m_CalcMeanGain);
    declProp("Length", m_length);
    declProp("OutputEDMPath", m_output_edm_path);
  
    m_charge_factor = m_cnv_unit;

#ifndef BUILD_ONLINE
    m_evt_id = -1;
#endif
}

WFSampling::~WFSampling()
{
}

bool WFSampling::initialize()
{
    if (m_rise_step <= 0) {
        LogError << "Invalid RisingEdgeStep (" << m_rise_step 
                 << "). It MUST be >= 1! Please check your python configuration." 
                 << std::endl;
        return false; 
    }
    // Handle the output EDM path
    if (m_output_edm_path.empty()) {
        m_output_edm_path = JM::Calib::CdLpmt::Path;
    }
    LogInfo << "Output EDM Path: " << m_output_edm_path << std::endl;
    LogInfo << "Decon Mode Status: " << (m_decon_mode ? "ON" : "OFF") << std::endl;

    // 获取 PMTParamSvc
    SniperPtr<IPMTParamSvc> pmtParamSvc(getParent(), "PMTParamSvc");
    if (pmtParamSvc.invalid()) {
        LogError << "Failed to get PMTParamSvc instance!" << std::endl;
        return false;
    }

    // 获取 PMTCalibSvc
    SniperPtr<IPMTCalibSvc> calSvc(getParent(), "PMTCalibSvc");
    if (calSvc.invalid()) {
        LogError << "Failed to get PMTCalibSvc instance!" << std::endl;
        return false;
    }
    m_pmt_calib_svc = calSvc.data();

    // 缓存 PMT 类型
    hmmtpmt.reserve(m_totalPMT);
    for (int i = 0; i < m_totalPMT; i++) {
        bool isHama = pmtParamSvc->isHamamatsu(i);
        hmmtpmt.push_back(isHama);
    }

    LogInfo << "WFSampling is initialized." << std::endl;

    return true;
}

bool WFSampling::execute()
{
    SniperDataPtr<JM::NavBuffer> navBuf(getParent(), "/Event");
    if (navBuf.invalid()) {
        LogError << "cannot get the NavBuffer @ /Event" << std::endl;
        return false;
    }
    m_buf = navBuf.data();
    auto nav = m_buf->curEvt();
    if (!nav) {
        LogWarn << "can't load event navigator." << std::endl;
        return dynamic_cast<Task*>(getRoot())->stop();
    }

    const std::map<int, JM::ElecWaveform*>* feeChannelsPtr = nullptr;

    std::string wf_path = m_decon_mode ? "/Event/CdDeconWaveform" : "/Event/CdWaveform";

    auto eh = JM::getHeaderObject<JM::CdWaveformHeader>(nav,wf_path);
    
    if (eh && eh->hasEvent()) {
        auto ee = eh->event();
        feeChannelsPtr = &(ee->channelData());
    } else if(!m_decon_mode){
        auto eh_fft = JM::getHeaderObject<JM::CdWaveformHeader>(nav, "/Event/CdWaveformOECFFT");
        if (eh_fft && eh_fft->hasEvent()) {
            auto ee_fft = eh_fft->event();
            feeChannelsPtr = &(ee_fft->channelData());
        }
    }

    if (feeChannelsPtr == nullptr) {
        return true;
    }
    
    const auto& feeChannels = *feeChannelsPtr;
    std::list<JM::CalibPmtChannel*> cpcl; 

    recChannels(feeChannels, cpcl); 
  
    auto ce = new JM::CdLpmtCalibEvt;
    ce->setCalibPMTCol(cpcl);
    
    auto ch = new JM::CdLpmtCalibHeader;
    ch->setEvent(ce);
  
    JM::addHeaderObject(nav, ch, m_output_edm_path);
    return true;
}

bool WFSampling::finalize()
{
    LogInfo << "Total baseline errors: " << m_stat << std::endl;
    return true;
}

bool WFSampling::recChannels(const std::map<int, JM::ElecWaveform*> & channels, std::list<JM::CalibPmtChannel*>& cpcl) 
{
    auto& gain = m_pmt_calib_svc->getGain();
    auto& gain_mean = m_pmt_calib_svc->getMeanGain();
    auto& toffset = m_pmt_calib_svc->getTimeOffset();

    for (const auto& it : channels) {
        const auto& channel = *(it.second);
        if (channel.adc().empty()) continue;

        unsigned int detID = -1;
        int pmtID = -1;

        if (it.first < m_totalPMT) { 
            pmtID = it.first;
            detID = CdID::id(pmtID, 0);
        } else { 
            detID = it.first;
            pmtID = CdID::module(Identifier(detID)); 
        }

        if (pmtID >= m_totalPMT) continue;

        const auto &adc_int = channel.adc();
        std::vector<float> adc_points; 
        std::vector<float> t_points;         

        // 根据 PMT 类型选择对应的阈值
        float current_thr_h = hmmtpmt.at(pmtID) ? m_thrh_h : m_thrh_n;
        int current_thr_w   = hmmtpmt.at(pmtID) ? m_thrw_h : m_thrw_n;

        if (m_decon_mode) {
            // ================= decon_mode =================
            // 没有找基线，没有高低增益，不翻转波形
            std::vector<float> AC;
            size_t limit = std::min(adc_int.size(), static_cast<size_t>(m_length));
            AC.reserve(limit);
            
            // 直接执行 (adc - 1000) / 100.0
            for (size_t i = 0; i < limit; ++i) {
                AC.push_back((static_cast<float>(adc_int[i]) - 1000.0f) / 100.0f);
            }

            // 直接丢给状态机找点
            extractKeyPoints(AC, m_thr, current_thr_h, current_thr_w, m_rise_step, t_points, adc_points);
            
            if (adc_points.empty()) continue;

        } else {
            // ================= normal_mode =================
            auto wf = channel.adc();
            std::vector<uint16_t> adc_int_new;
            adc_int_new.reserve(adc_int.size());

            if (m_polarity < 0) {
                for (auto val : adc_int) {
                    adc_int_new.push_back(16384 - val);
                }
            } else {
                adc_int_new = adc_int;
            }

            std::vector<float> baseline(channel.adc().size());
            float rms = 0.0f;

            // 根据量程填充波形
            if (channel.isLowRange()) {
                for (size_t i = 0; i < static_cast<size_t>(m_length) && i < wf.size(); ++i) wf[i] = adc_int_new.at(i);
            } else {
                for (size_t i = 0; i < static_cast<size_t>(m_length) && i < wf.size(); ++i) wf[i] = adc_int_new.at(i) * m_cnv_high_range;
            }

            // 执行带有找基线和计算AC的旧逻辑
            getREC1(wf, adc_points, t_points, baseline, rms, current_thr_h, current_thr_w, m_rise_step);
            if (adc_points.empty()) continue;
        }

        // =================== Log 打印块 ===================
        float first_t = t_points.front();
        float first_adc = adc_points.front();
        float first_peak_t = first_t;
        float first_peak_adc = first_adc;
        
        for (size_t i = 0; i < adc_points.size() - 1; ++i) {
            if (adc_points[i] > adc_points[i+1]) {
                first_peak_t = t_points[i];
                first_peak_adc = adc_points[i];
                break;
            }
        }
        
        if (adc_points.size() == 1 || (first_peak_t == first_t && adc_points.back() > first_adc)) {
            first_peak_t = t_points.back();
            first_peak_adc = adc_points.back();
        }

        LogDebug << "PMT ID: " << pmtID 
                 << " | Extracted Points: " << t_points.size() 
                 << " | First Point: (t=" << first_t << ", adc=" << first_adc << ")"
                 << " | First Peak: (t=" << first_peak_t << ", adc=" << first_peak_adc << ")" 
                 << std::endl;
        // =========================================================

        // 仅在 Normal 模式下除以 Gain (Decon 模式跳过该步骤)
        if (!m_decon_mode) {
            for (size_t i = 0; i < adc_points.size(); ++i) {
                if (m_CalcMeanGain == CalcMeanGain::FitSpectral && gain_mean.at(pmtID) > 0) {
                    adc_points.at(i) /= gain_mean.at(pmtID);
                } else if (m_CalcMeanGain == CalcMeanGain::NormPeakGain && gain.at(pmtID) > 0) {
                    adc_points.at(i) /= gain.at(pmtID);
                }
            }
        }

        // 时间仍然统一减去 time offset
        for (size_t i = 0; i < t_points.size(); ++i) {
            t_points.at(i) -= toffset.at(pmtID);
        }

        float firstHitTime = t_points.front();

        // 构造 EDM Output
        auto cpc = new JM::CalibPmtChannel;
        cpc->setPmtId(detID);
        cpc->setFirstHitTime(firstHitTime);
        cpc->setTime(t_points);
        cpc->setCharge(adc_points); 
        cpcl.push_back(cpc);
    }
    return true;
}

double WFSampling::getREC1(std::vector<uint16_t>& adc, std::vector<float>& adc_points, std::vector<float>& t_points, std::vector<float>& baseline, float& rms, float thr_h, int thr_w, int rise_step)
{
    adc_points.clear();
    t_points.clear();
    baseline.resize(adc.size());
    std::fill(baseline.begin(), baseline.end(), -1.0f);
  
    // 1. 计算静态基线
    float baseline_piece = 0.0f;
    std::vector<uint16_t> adc_piece;
    std::vector<uint16_t> adc_bl;
    
    int start = 0;
    const int piece = 8;
    bool flag_good = true;
    
    for (start = 0; start < m_bl - 3 * piece; start += piece) {
        flag_good = true;
        adc_piece.clear();
        for (int i = 0; i < piece; ++i) {
            adc_piece.push_back(adc[i + start]);
        }
        
        baseline_piece = GetMean(adc_piece);
        for (int i = 0; i < piece; ++i) {
            if (std::abs(static_cast<float>(adc_piece[i]) - baseline_piece) > m_thr_flat) {
                flag_good = false;
                break;
            }
        }
        
        if (flag_good) break;
    }
  
    if (!flag_good) {
        m_stat++;
        return -1000.0;
    }
  
    baseline_piece = GetMean(adc_piece);
    for (int i = start; i < m_bl; ++i) {
        if (std::abs(static_cast<float>(adc[i]) - baseline_piece) > m_thr_flat) break;
        adc_bl.push_back(adc[i]);
    }
  
    std::fill(baseline.begin(), baseline.end(), GetMean(adc_bl));
    rms = GetRMS(adc_bl, baseline.at(0));

    // 2. 准备物理波形 AC (扣除基线并修正极性)
    std::vector<float> AC(adc.size(), 0.0f);
    for (size_t i = 0; i < adc.size(); ++i) {
        if (baseline[i] >= 0.0f) {
            AC[i] = adc[i] - baseline[i];
        }
    }

    // 3. 设定动态的 Global Threshold
    float global_thr = m_thr;

    // 4. 调用核心算法
    extractKeyPoints(AC, global_thr, thr_h, thr_w, rise_step, t_points, adc_points);

    return 0.0; 
}

void WFSampling::extractKeyPoints(const std::vector<float>& AC, float global_thr, float h_thr, int w_thr, int rise_step, std::vector<float>& out_time, std::vector<float>& out_adc) 
{
    out_time.clear();
    out_adc.clear();
    if (AC.empty()) return;

    int n = AC.size();
    bool in_pulse = false;           
    bool looking_for_peak = false;   
    
    // 用于计算宽度的前一个有效谷底
    int last_valley_time = 0;
    float last_valley_adc = 0.0f;
    
    // 迟滞包络跟踪器
    float local_max_adc = -1e9;
    int local_max_time = -1;
    float local_min_adc = 1e9;
    int local_min_time = -1;
    
    int confirmed_peak_time = -1;
    float confirmed_peak_adc = -1.0f;
    
    // 1. 定义一个安全的“守门员” Lambda 函数
    auto add_point = [&](int t, float val) {
        if (out_time.empty() || static_cast<float>(t) > out_time.back()) {
            out_time.push_back(static_cast<float>(t));
            out_adc.push_back(val);
        }
    };

    // 2. 记录是否已经对第一个峰的上升沿进行过采样
    bool first_peak_sampled = false;

    for (int i = 1; i < n - 1; ++i) {
        float curr = AC[i];

        if (curr > global_thr) {
            if (!in_pulse) {
                in_pulse = true;
                looking_for_peak = true; 
                
                last_valley_time = i;
                last_valley_adc = curr;
                
                local_max_adc = curr;
                local_max_time = i;
                local_min_adc = curr;
                local_min_time = i;
            }

            if (curr > local_max_adc) {
                local_max_adc = curr;
                local_max_time = i;
            }
            if (curr < local_min_adc) {
                local_min_adc = curr;
                local_min_time = i;
            }

            if (looking_for_peak) {
                if (curr < local_max_adc - h_thr) { 
                    confirmed_peak_time = local_max_time;
                    confirmed_peak_adc = local_max_adc;
                    
                    looking_for_peak = false; 
                    local_min_adc = curr;
                    local_min_time = i;
                }
            } else {
                if (curr > local_min_adc + h_thr) {
                    int width = local_min_time - last_valley_time;

                    if (width >= w_thr) {
                        add_point(last_valley_time, last_valley_adc);
                        if (!first_peak_sampled) {
                            for (int t = last_valley_time + rise_step; t < confirmed_peak_time; t += rise_step) {
                                add_point(t, AC[t]);
                            }
                            first_peak_sampled = true; 
                        }
                        
                        add_point(confirmed_peak_time, confirmed_peak_adc);
                        add_point(local_min_time, local_min_adc);

                        last_valley_time = local_min_time;
                        last_valley_adc = local_min_adc;
                        local_max_adc = curr;
                        local_max_time = i;
                    }
                    
                    looking_for_peak = true; 
                }
            }
        } 
        else {
            if (in_pulse) {
                in_pulse = false;
                
                if (!looking_for_peak) { 
                    int width = i - last_valley_time;
                    if (width >= w_thr) {
                        add_point(last_valley_time, last_valley_adc);
                        if (!first_peak_sampled) {
                            for (int t = last_valley_time + rise_step; t < confirmed_peak_time; t += rise_step) {
                                add_point(t, AC[t]);
                            }
                            first_peak_sampled = true;
                        }
                        add_point(confirmed_peak_time, confirmed_peak_adc);
                        add_point(i, curr);
                    }
                } else {
                    if (local_max_adc - last_valley_adc >= h_thr) {
                        int width = i - last_valley_time;
                        if (width >= w_thr) {
                            add_point(last_valley_time, last_valley_adc);
                            if (!first_peak_sampled) {
                                for (int t = last_valley_time + rise_step; t < local_max_time; t += rise_step) {
                                    add_point(t, AC[t]);
                                }
                                first_peak_sampled = true;
                            }
                            add_point(local_max_time, local_max_adc);
                            add_point(i, curr);
                        }
                    }
                }
            }
        }
    }
}

float WFSampling::GetMean(const std::vector<uint16_t> &adc) {
    if (adc.empty()) return 0.0f;
    float result = 0.0f;
    for (auto val : adc) {
        result += static_cast<float>(val);
    }
    return result / static_cast<float>(adc.size());
}

float WFSampling::GetRMS(const std::vector<uint16_t> &adc, float baseline) {
    if (adc.empty()) return 0.0f;
    if (baseline < 0.0f) baseline = GetMean(adc);
    
    float result = 0.0f;
    float size_f = static_cast<float>(adc.size());
    for (auto val : adc) {
        float diff = static_cast<float>(val) - baseline;
        result += (diff * diff) / size_f;
    }
    return std::sqrt(result);
}

#endif