#ifndef WFSampling_cc
#define WFSampling_cc

// ---------------- C++ Standard Library ----------------
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>


// ---------------- ROOT Library ----------------
#include <TTimeStamp.h>
#include <TSystem.h>
#include "TROOT.h"
#include <TTree.h>
#include "Rtypes.h" // <=== 显式引入以确保 float 可用

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
    m_decon_mode(false), 
    m_CalcMeanGain(CalcMeanGain::FitSpectral),
    m_output_edm_path(""),
    m_c(1),
    m_memMgr(nullptr), 
    m_alg("stt"), 
    m_stat(0),
    m_process_cd(true),
    m_enableWaveformOutput(false),
    m_waveformTree(nullptr),
    m_TotalKeyPoints(0),
    m_MaxKeyPointsPerPMT(0),
    m_AvgKeyPointsPerPMT(0),
    m_ActivePMTs(0)
{
    declProp("TotalPMT", m_totalPMT);
    declProp("BslBufLen", m_bl);
    declProp("Threshold", m_thr);
    declProp("Threshold_FHT", m_thr_fht);
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
    declProp("DeconMode", m_decon_mode); 
    declProp("CalcMeanGain", m_CalcMeanGain);
    declProp("EnableWaveformOutput", m_enableWaveformOutput);
    declProp("Length", m_length);
    declProp("OutputEDMPath", m_output_edm_path);
  
    m_charge_factor = m_cnv_unit;

    m_evt_id = 0;
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


#ifndef BUILD_ONLINE
    if(m_enableUserOutput || m_enableWaveformOutput){

        SniperPtr<RootWriter> svc(getParent(), "RootWriter");

        if (svc.invalid()){
            LogError << "Can't Locate RootWriter. If you want to use it, please enalbe it in your job option file." << std::endl;
            return false;
        }
        
        if (m_enableUserOutput){
            m_calib = svc->bookTree(*m_par, "USER_OUTPUT/calibevt", "event statistics & hit details");
            m_calib->Branch("EventID",          &m_evt_id,          "EventID/I");
            m_calib->Branch("TotalKeyPoints",   &m_TotalKeyPoints,   "TotalKeyPoints/I");
            m_calib->Branch("MaxKeyPointsPerPMT",&m_MaxKeyPointsPerPMT,"MaxKeyPointsPerPMT/I");
            m_calib->Branch("AvgKeyPointsPerPMT",&m_AvgKeyPointsPerPMT,"AvgKeyPointsPerPMT/F");
            m_calib->Branch("ActivePMTs",       &m_ActivePMTs,       "ActivePMTs/I");
            m_calib->Branch("Charge",           &m_charge);          
            m_calib->Branch("Time",             &m_time);
            m_calib->Branch("PMTID",            &m_pmtid);
            m_calib->Branch("TrigTimeSec",      &m_trigtimesec,     "TrigTimeSec/I");
            m_calib->Branch("TrigTimeNanoSec",  &m_trigtimenanosec, "TrigTimeNanoSec/I");
        }

        // ===== 2. per‑Event 波形树 (waveform) =====
        if (m_enableWaveformOutput) {
            m_waveformTree = svc->bookTree(*m_par, "USER_OUTPUT/waveform", "waveform for threshold tuning");
            m_waveformTree->Branch("EventID",         &m_evt_id,         "EventID/I");
            m_waveformTree->Branch("PMTID",           &m_wfPMTID);       
            gInterpreter->GenerateDictionary("vector<vector<float>>", "vector");
            m_waveformTree->Branch("ADC",             &m_wfADC);         // 此时为 vector<vector<float>> 分支
            m_waveformTree->Branch("TrigTimeSec",     &m_trigtimesec,     "TrigTimeSec/I");
            m_waveformTree->Branch("TrigTimeNanoSec", &m_trigtimenanosec, "TrigTimeNanoSec/I");
        }
    }
#endif

    LogInfo << "WFSampling is initialized." << std::endl;

    return true;
}

bool WFSampling::execute()
{
    if(m_enableUserOutput){
        m_charge.clear();
        m_time.clear();
        m_pmtid.clear();
        m_TotalKeyPoints = 0;
        m_MaxKeyPointsPerPMT = 0;
        m_AvgKeyPointsPerPMT = 0;
        m_ActivePMTs = 0;
    }
    
    if(m_enableWaveformOutput){
        m_wfPMTID.clear();
        m_wfADC.clear();
    }

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

    const TTimeStamp& ts = nav->TimeStamp();

    m_trigtimesec = ts.GetSec();
    m_trigtimenanosec = ts.GetNanoSec();

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

#ifndef BUILD_ONLINE
    if(m_enableUserOutput){
        if (m_ActivePMTs > 0) {
            m_AvgKeyPointsPerPMT = static_cast<float>(m_TotalKeyPoints) / m_ActivePMTs;
        } else m_AvgKeyPointsPerPMT = 0.0f;

        m_calib->Fill();
    }
    
    if(m_enableWaveformOutput){
        m_waveformTree->Fill();
    }
    
    if(m_enableUserOutput || m_enableWaveformOutput){
        m_evt_id++;
    }
#endif

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
            std::vector<float> AC;
            size_t limit = std::min(adc_int.size(), static_cast<size_t>(m_length));
            AC.reserve(limit);
            
            for (size_t i = 0; i < limit; ++i) {
                AC.push_back((static_cast<float>(adc_int[i]) - 1000.0f) / 100.0f);
            }

            extractKeyPoints(AC, m_thr, current_thr_h, current_thr_w, m_rise_step, t_points, adc_points);

#ifndef BUILD_ONLINE
            // === Decon 模式的波形缓存 (采用 float) ===
            if (m_enableWaveformOutput) {
                std::vector<float> ac_wf;
                ac_wf.reserve(AC.size());
                for (float val : AC) {
                    ac_wf.push_back(val);
                }
                m_wfPMTID.push_back(pmtID);
                m_wfADC.push_back(ac_wf);
            }
#endif
        } else {
            // ================= normal_mode =================
            std::vector<float> wf(channel.adc().size());
            std::vector<uint16_t> adc_int_new;
            adc_int_new.reserve(adc_int.size());

            if (m_polarity < 0) {
                for (auto val : adc_int) {
                    int inverted_val = 16384 - static_cast<int>(val);
                    if (inverted_val < 0) {
                        inverted_val = 0;
                    }
                    adc_int_new.push_back(static_cast<uint16_t>(inverted_val));
                }
            } else {
                adc_int_new = adc_int;
            }

            std::vector<float> baseline(channel.adc().size());
            float rms = 0.0f;

            if (channel.isLowRange()) {
                for (size_t i = 0; i < static_cast<size_t>(m_length) && i < wf.size(); ++i) wf[i] = adc_int_new.at(i);
            } else {
                for (size_t i = 0; i < static_cast<size_t>(m_length) && i < wf.size(); ++i) wf[i] = adc_int_new.at(i) * m_cnv_high_range;
            }


            getREC1(wf, adc_points, t_points, baseline, rms, current_thr_h, current_thr_w, m_rise_step, pmtID);
        }
        if (adc_points.empty()) continue;

        if (m_enableUserOutput) {
            int nPoints = static_cast<int>(adc_points.size());
            m_TotalKeyPoints += nPoints;
            m_ActivePMTs++;
            if (nPoints > m_MaxKeyPointsPerPMT) {
                m_MaxKeyPointsPerPMT = nPoints;
            }
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
        for (size_t i = 0; i < adc_points.size(); ++i) {
            if(!m_decon_mode){
                if (m_CalcMeanGain == CalcMeanGain::FitSpectral && gain_mean.at(pmtID) > 0) {
                    adc_points.at(i) /= gain_mean.at(pmtID);
                } else if (m_CalcMeanGain == CalcMeanGain::NormPeakGain && gain.at(pmtID) > 0) {
                    adc_points.at(i) /= gain.at(pmtID);
                }
            }
            t_points.at(i) -= toffset.at(pmtID);

#ifndef BUILD_ONLINE
            if (m_enableUserOutput){
                m_charge.push_back(adc_points[i]);
                m_time.push_back(t_points[i]);
                m_pmtid.push_back(pmtID);
            }
#endif
        }

        float firstHitTime = t_points.front();

        auto cpc = new JM::CalibPmtChannel;
        cpc->setPmtId(detID);
        cpc->setFirstHitTime(firstHitTime);
        cpc->setTime(t_points);
        cpc->setCharge(adc_points); 
        cpcl.push_back(cpc);
    }
    return true;
}

double WFSampling::getREC1(std::vector<float>& adc, std::vector<float>& adc_points, std::vector<float>& t_points, std::vector<float>& baseline, float& rms, float thr_h, int thr_w, int rise_step, int pmtID)
{
    adc_points.clear();
    t_points.clear();
    baseline.resize(adc.size());
    std::fill(baseline.begin(), baseline.end(), -1.0f);
  
    // 1. 计算静态基线
    float baseline_piece = 0.0f;
    std::vector<float> adc_piece;
    std::vector<float> adc_bl;
    
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

    // 2. 准备物理波形 AC 
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

#ifndef BUILD_ONLINE
    if (m_enableWaveformOutput) {
        std::vector<float> ac_wf;
        ac_wf.reserve(AC.size());
        for (float val : AC) {
            ac_wf.push_back(val);
        }
        m_wfPMTID.push_back(pmtID);
        m_wfADC.push_back(ac_wf);
    }
#endif

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
    
    int last_valley_time = 0;
    float last_valley_adc = 0.0f;
    
    float local_max_adc = std::numeric_limits<float>::lowest();
    int local_max_time = -1;
    float local_min_adc = std::numeric_limits<float>::max();
    int local_min_time = -1;
    
    int confirmed_peak_time = -1;
    float confirmed_peak_adc = -1.0f;
        
    auto add_point = [&](int t, float val) {
        if (out_time.empty() || static_cast<float>(t) > out_time.back()) {
            out_time.push_back(static_cast<float>(t));
            out_adc.push_back(val);
        }
    };

    bool first_peak_sampled = false;

    float threshold_fht;
    auto max_AC = std::max_element(AC.begin(), AC.end());
    float peak_value = *max_AC;

    if (m_thr_fht >= 1 ){
        threshold_fht = m_thr_fht;
    }
    else {
        threshold_fht = m_thr_fht * peak_value;
    }

    float thr_slope = peak_value * 0.55f;
    bool found_fht = false;

    // 【重构】提取公共逻辑：专门处理第一个峰的步长采样与 55% 阈值点采样
    auto sample_first_peak = [&](int start_t, int end_t) {
        
        // 为了防止 55% 的点由于时间戳乱序被 add_point 拦截，我们用临时向量打包这一批点
        std::vector<std::pair<int, float>> temp_points;
        
        // 1. 步长采样
        for (int t = start_t + rise_step; t < end_t; t += rise_step) {
            temp_points.push_back({t, AC[t]});
        }
        
        // 2. 55% 阈值点采样
        int idx_55 = start_t;
        float best_diff = std::abs(AC[start_t] - thr_slope);
        for (int t = start_t + 1; t <= end_t; ++t) {
            float diff = std::abs(AC[t] - thr_slope);
            if (diff < best_diff) {
                best_diff = diff;
                idx_55 = t;
            }
        }
        temp_points.push_back({idx_55, AC[idx_55]});
        
        // 3. 按时间戳排序并去重（避免 idx_55 与步长采样点重合导致重复）
        std::sort(temp_points.begin(), temp_points.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        temp_points.erase(std::unique(temp_points.begin(), temp_points.end(), [](const auto& a, const auto& b) {
            return a.first == b.first;
        }), temp_points.end());
        
        // 4. 安全地送入打点器
        for (const auto& pt : temp_points) {
            add_point(pt.first, pt.second);
        }
        
        first_peak_sampled = true;
    };

    // 主循环
    for (int i = 1; i < n - 1; ++i) {
        float curr = AC[i];
        if (curr > threshold_fht && !found_fht){
            found_fht = true;
        }
        
        if (curr > global_thr && found_fht) {
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
                        sample_first_peak(last_valley_time, confirmed_peak_time); 
                        
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
                        if(!first_peak_sampled){
                            sample_first_peak(last_valley_time, confirmed_peak_time); // 调用公共Lambda
                        }
                        add_point(confirmed_peak_time, confirmed_peak_adc);
                        add_point(i, curr);
                    }
                } else {
                    if (local_max_adc - last_valley_adc >= h_thr) {
                        int width = i - last_valley_time;
                        if (width >= w_thr) {
                            add_point(last_valley_time, last_valley_adc);
                            if(!first_peak_sampled){
                                sample_first_peak(last_valley_time, confirmed_peak_time); // 调用公共Lambda
                            }
                            add_point(local_max_time, local_max_adc);
                            add_point(i, curr);
                        }
                    }
                }
            }
        }
    }

    // ==========================================
    // 【重要修复】收尾逻辑：处理触及信号末尾且未闭合的脉冲
    // ==========================================
    if (in_pulse) {
        int last_idx = n - 2; // 对应循环中可能达到的最大合法索引
        float last_curr = AC[last_idx];

        if (!looking_for_peak) { 
            int width = last_idx - last_valley_time;
            if (width >= w_thr) {
                add_point(last_valley_time, last_valley_adc);
                sample_first_peak(last_valley_time, confirmed_peak_time);
                add_point(confirmed_peak_time, confirmed_peak_adc);
                add_point(last_idx, last_curr);
            }
        } else {
            if (local_max_adc - last_valley_adc >= h_thr) {
                int width = last_idx - last_valley_time;
                if (width >= w_thr) {
                    add_point(last_valley_time, last_valley_adc);
                    if(!first_peak_sampled){
                        sample_first_peak(last_valley_time, confirmed_peak_time); // 调用公共Lambda
                    }
                    add_point(local_max_time, local_max_adc);
                    add_point(last_idx, last_curr);
                }
            }
        }
    }
}

float WFSampling::GetMean(const std::vector<float> &adc) {
    if (adc.empty()) return 0.0f;
    float result = 0.0f;
    for (auto val : adc) {
        result += static_cast<float>(val);
    }
    return result / static_cast<float>(adc.size());
}

float WFSampling::GetRMS(const std::vector<float> &adc, float baseline) {
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