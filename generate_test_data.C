/**
 * generate_test_data.C
 * ============================================================
 * Generates a synthetic ROOT file simulating a PET detector array.
 *
 * Produces test_detector_data.root containing:
 *   TTree  "detector_data"   branches: channel_id, adc_value, timestamp
 *   TH1F   "spectrum_ch0000" ... "spectrum_ch0015"  (pre-filled histograms)
 *
 * Simulated peaks (Na-22 source — standard PET calibration):
 *   511.0  keV  annihilation peak   (70% of signal events)
 *   1274.5 keV  Na-22 gamma peak    (30% of signal events)
 *
 * Each channel has a unique gain and offset to simulate real detector
 * response variation across the array.
 *
 * Usage (ROOT prompt):
 *   root -l -q generate_test_data.C
 *
 * Or compiled:
 *   root -l -b -q generate_test_data.C+
 * ============================================================
 */

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TRandom3.h"
#include "TMath.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>   // std::clamp (C++17)

// ── Configuration ─────────────────────────────────────────────────────── //
static const int    N_CHANNELS        = 16;
static const int    EVENTS_PER_CHAN   = 200000;
static const int    N_BINS            = 1024;
static const double ADC_MIN           = 0.0;
static const double ADC_MAX           = 4096.0;
static const double BG_FRACTION       = 0.30;   // Compton background fraction
static const double FWHM_FRAC_511     = 0.12;   // 12% FWHM at 511 keV
static const char*  OUTPUT_FILENAME   = "test_detector_data.root";

// Peak energies and relative intensities
static const int    N_PEAKS           = 2;
static const double PEAK_ENERGY[N_PEAKS]     = { 511.0,  1274.5 };
static const double PEAK_INTENSITY[N_PEAKS]  = { 0.70,   0.30   };

// ── Helper functions ───────────────────────────────────────────────────── //

// Energy resolution: sigma in keV (scales as 1/sqrt(E))
double sigmaKeV(double energy_keV)
{
    double fwhm = energy_keV * FWHM_FRAC_511 * TMath::Sqrt(511.0 / energy_keV);
    return fwhm / 2.355;
}

// Convert true energy to ADC channel (linear calibration)
double energyToADC(double energy_keV, double gain, double offset)
{
    return (energy_keV - offset) / gain;
}

// ── Main macro ─────────────────────────────────────────────────────────── //
void generate_test_data()
{
    TRandom3 rng(42);   // fixed seed for reproducibility

    // ── Per-channel true calibration parameters ──────────────────────── //
    // Gain  : ADC counts per keV  (varies channel to channel)
    // Offset: ADC pedestal offset
    double trueGain[N_CHANNELS];
    double trueOffset[N_CHANNELS];

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        trueGain[ch]   = rng.Uniform(0.45, 0.55);   // keV / ADC count
        trueOffset[ch] = rng.Uniform(-15.0, 15.0);  // ADC offset
    }

    // ── Open output file ─────────────────────────────────────────────── //
    TFile* f = TFile::Open(OUTPUT_FILENAME, "RECREATE");
    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: Cannot create " << OUTPUT_FILENAME << std::endl;
        return;
    }

    // ── 1. TTree ─────────────────────────────────────────────────────── //
    std::cout << "Generating TTree..." << std::endl;

    TTree* tree = new TTree("detector_data", "Simulated PET detector data");

    Int_t    channel_id;
    Float_t  adc_value;
    Double_t timestamp;

    tree->Branch("channel_id", &channel_id, "channel_id/I");
    tree->Branch("adc_value",  &adc_value,  "adc_value/F");
    tree->Branch("timestamp",  &timestamp,  "timestamp/D");

    double cumTime = 0.0;

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        for (int ev = 0; ev < EVENTS_PER_CHAN; ev++) {

            double measuredE;

            // Background or signal?
            if (rng.Uniform() < BG_FRACTION) {
                // Flat Compton background
                measuredE = rng.Uniform(50.0, 1350.0);
            } else {
                // Pick a peak by intensity
                double r = rng.Uniform();
                double cumulative = 0.0;
                int peakIdx = N_PEAKS - 1;
                for (int p = 0; p < N_PEAKS; p++) {
                    cumulative += PEAK_INTENSITY[p];
                    if (r < cumulative) { peakIdx = p; break; }
                }
                double trueE = PEAK_ENERGY[peakIdx];
                measuredE = rng.Gaus(trueE, sigmaKeV(trueE));
            }

            // ADC conversion + electronic noise
            double adc = energyToADC(measuredE, trueGain[ch], trueOffset[ch]);
            adc += rng.Gaus(0.0, 1.5);   // ADC noise
            adc  = std::clamp(adc, ADC_MIN, ADC_MAX - 1.0);

            // Random exponential inter-arrival time (Poisson process)
            cumTime += rng.Exp(1.0e-7);

            channel_id = ch;
            adc_value  = static_cast<Float_t>(adc);
            timestamp  = cumTime;
            tree->Fill();
        }

        if ((ch + 1) % 4 == 0)
            std::cout << "  TTree: " << ch + 1 << "/" << N_CHANNELS
                      << " channels done" << std::endl;
    }

    tree->Write();

    // ── 2. TH1F histograms ───────────────────────────────────────────── //
    std::cout << "Generating TH1F histograms..." << std::endl;

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        char hname[64], htitle[128];
        std::snprintf(hname,  sizeof(hname),  "spectrum_ch%04d", ch);
        std::snprintf(htitle, sizeof(htitle),
                      "Channel %d spectrum;ADC value;Counts", ch);

        TH1F* h = new TH1F(hname, htitle, N_BINS, ADC_MIN, ADC_MAX);
        h->SetDirectory(f);

        // Re-seed per channel so histograms match TTree content
        TRandom3 rng2(42 + ch * 1000);

        for (int ev = 0; ev < EVENTS_PER_CHAN; ev++) {
            double measuredE;
            if (rng2.Uniform() < BG_FRACTION) {
                measuredE = rng2.Uniform(50.0, 1350.0);
            } else {
                double r = rng2.Uniform();
                double cumulative = 0.0;
                int peakIdx = N_PEAKS - 1;
                for (int p = 0; p < N_PEAKS; p++) {
                    cumulative += PEAK_INTENSITY[p];
                    if (r < cumulative) { peakIdx = p; break; }
                }
                double trueE = PEAK_ENERGY[peakIdx];
                measuredE = rng2.Gaus(trueE, sigmaKeV(trueE));
            }
            double adc = energyToADC(measuredE, trueGain[ch], trueOffset[ch]);
            adc += rng2.Gaus(0.0, 1.5);
            adc  = std::clamp(adc, ADC_MIN, ADC_MAX - 1.0);
            h->Fill(adc);
        }

        h->Write();
    }

    f->Close();

    // ── Summary ──────────────────────────────────────────────────────── //
    std::cout << "\n✅  Written: " << OUTPUT_FILENAME << std::endl;
    std::cout << "   TTree   : 'detector_data'  ("
              << N_CHANNELS * EVENTS_PER_CHAN << " total events)" << std::endl;
    std::cout << "             Branches: channel_id, adc_value, timestamp\n";
    std::cout << "   TH1F    : spectrum_ch0000 ... spectrum_ch"
              << std::setfill('0') << std::setw(4) << N_CHANNELS - 1 << "\n";

    std::cout << "\nTrue calibration parameters (for result verification):\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(6)  << "Ch"
              << std::setw(18) << "Gain (keV/ADC)"
              << std::setw(16) << "Offset (ADC)" << "\n";
    std::cout << std::string(40, '-') << "\n";
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        std::cout << std::setw(6)  << ch
                  << std::setw(18) << trueGain[ch]
                  << std::setw(16) << trueOffset[ch] << "\n";
    }

    std::cout << "\nExpected peak ADC positions (first 4 channels):\n";
    for (int ch = 0; ch < 4; ch++) {
        std::cout << "  Ch " << ch << ":";
        for (int p = 0; p < N_PEAKS; p++) {
            double adcPos = energyToADC(PEAK_ENERGY[p], trueGain[ch], trueOffset[ch]);
            std::cout << "  " << PEAK_ENERGY[p] << " keV → ADC≈"
                      << std::setprecision(1) << adcPos;
        }
        std::cout << "\n";
    }
    std::cout << "\nVerify these match your fitted calibration output.\n";
}
