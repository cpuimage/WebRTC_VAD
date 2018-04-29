/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


/*
 * This header file includes the VAD API calls. Specific function calls are given below.
 */

#ifndef COMMON_AUDIO_VAD_INCLUDE_WEBRTC_VAD_H_  // NOLINT
#define COMMON_AUDIO_VAD_INCLUDE_WEBRTC_VAD_H_

#include <stddef.h>
#include <stdint.h>  // NOLINT(build/include)
#include <string.h>
#include <assert.h>
// If you for some reson need to know if DCHECKs are on, test the value of
// RTC_DCHECK_IS_ON. (Test its value, not if it's defined; it'll always be
// defined, to either a true or a false value.)
#if !defined(NDEBUG) || defined(DCHECK_ALWAYS_ON)
#define RTC_DCHECK_IS_ON 1
#else
#define RTC_DCHECK_IS_ON 0
#endif

// C version. Lacks many features compared to the C++ version, but usage
// guidelines are the same.

#define RTC_DCHECK(condition)   assert(condition)
#define RTC_DCHECK_LE(a, b) RTC_DCHECK((a) <= (b))
#define RTC_DCHECK_LT(a, b) RTC_DCHECK((a) < (b))
#define RTC_DCHECK_GT(a, b) RTC_DCHECK((a) > (b))

int32_t WebRtcSpl_Energy(int16_t *vector,
                         size_t vector_length,
                         int *scale_factor);

enum {
    kNumChannels = 6
};  // Number of frequency bands (named channels).
enum {
    kNumGaussians = 2
};  // Number of Gaussians per channel in the GMM.
enum {
    kTableSize = kNumChannels * kNumGaussians
};
enum {
    kMinEnergy = 10
};  // Minimum energy required to trigger audio signal.

typedef struct VadInstT_ {
    int vad;
    int32_t downsampling_filter_states[4];
    int16_t noise_means[kTableSize];
    int16_t speech_means[kTableSize];
    int16_t noise_stds[kTableSize];
    int16_t speech_stds[kTableSize];
    // TODO(bjornv): Change to |frame_count|.
    int32_t frame_counter;
    int16_t over_hang;  // Over Hang
    int16_t num_of_speech;
    // TODO(bjornv): Change to |age_vector|.
    int16_t index_vector[16 * kNumChannels];
    int16_t low_value_vector[16 * kNumChannels];
    // TODO(bjornv): Change to |median|.
    int16_t mean_value[kNumChannels];
    int16_t upper_state[5];
    int16_t lower_state[5];
    int16_t hp_filter_state[4];
    int16_t over_hang_max_1[3];
    int16_t over_hang_max_2[3];
    int16_t individual[3];
    int16_t total[3];

    int init_flag;
} VadInstT;

// Initializes the core VAD component. The default aggressiveness mode is
// controlled by |kDefaultMode| in vad_core.c.
//
// - self [i/o] : Instance that should be initialized
//
// returns      : 0 (OK), -1 (null pointer in or if the default mode can't be
//                set)
int WebRtcVad_InitCore(VadInstT *self);

/****************************************************************************
 * WebRtcVad_set_mode_core(...)
 *
 * This function changes the VAD settings
 *
 * Input:
 *      - inst      : VAD instance
 *      - mode      : Aggressiveness degree
 *                    0 (High quality) - 3 (Highly aggressive)
 *
 * Output:
 *      - inst      : Changed  instance
 *
 * Return value     :  0 - Ok
 *                    -1 - Error
 */

int WebRtcVad_set_mode_core(VadInstT *self, int mode);

/****************************************************************************

 * WebRtcVad_CalcVad8khz(...)
 *
 * Calculate probability for active speech and make VAD decision.
 *
 * Input:
 *      - inst          : Instance that should be initialized
 *      - speech_frame  : Input speech frame
 *      - frame_length  : Number of input samples
 *
 * Output:
 *      - inst          : Updated filter states etc.
 *
 * Return value         : VAD decision
 *                        0 - No active speech
 *                        1-6 - Active speech
 */
int WebRtcVad_CalcVad8khz(VadInstT *inst, const int16_t *speech_frame,
                          size_t frame_length);


// Updates and returns the smoothed feature minimum. As minimum we use the
// median of the five smallest feature values in a 100 frames long window.
// As long as |handle->frame_counter| is zero, that is, we haven't received any
// "valid" data, FindMinimum() outputs the default value of 1600.
//
// Inputs:
//      - feature_value : New feature value to update with.
//      - channel       : Channel number.
//
// Input & Output:
//      - handle        : State information of the VAD.
//
// Returns:
//                      : Smoothed minimum value for a moving window.
int16_t WebRtcVad_FindMinimum(VadInstT *handle,
                              int16_t feature_value,
                              int channel);

// Calculates the probability for |input|, given that |input| comes from a
// normal distribution with mean and standard deviation (|mean|, |std|).
//
// Inputs:
//      - input         : input sample in Q4.
//      - mean          : mean input in the statistical model, Q7.
//      - std           : standard deviation, Q7.
//
// Output:
//
//      - delta         : input used when updating the model, Q11.
//                        |delta| = (|input| - |mean|) / |std|^2.
//
// Return:
//   (probability for |input|) =
//    1 / |std| * exp(-(|input| - |mean|)^2 / (2 * |std|^2));
int32_t WebRtcVad_GaussianProbability(int16_t input,
                                      int16_t mean,
                                      int16_t std,
                                      int16_t *delta);


// Takes |data_length| samples of |data_in| and calculates the logarithm of the
// energy of each of the |kNumChannels| = 6 frequency bands used by the VAD:
//        80 Hz - 250 Hz
//        250 Hz - 500 Hz
//        500 Hz - 1000 Hz
//        1000 Hz - 2000 Hz
//        2000 Hz - 3000 Hz
//        3000 Hz - 4000 Hz
//
// The values are given in Q4 and written to |features|. Further, an approximate
// overall energy is returned. The return value is used in
// WebRtcVad_GmmProbability() as a signal indicator, hence it is arbitrary above
// the threshold |kMinEnergy|.
//
// - self         [i/o] : State information of the VAD.
// - data_in      [i]   : Input audio data, for feature extraction.
// - data_length  [i]   : Audio data size, in number of samples.
// - features     [o]   : 10 * log10(energy in each frequency band), Q4.
// - returns            : Total energy of the signal (NOTE! This value is not
//                        exact. It is only used in a comparison.)
int16_t WebRtcVad_CalculateFeatures(VadInstT *self, const int16_t *data_in,
                                    size_t data_length, int16_t *features);


typedef struct WebRtcVadInst VadInst;

#ifdef __cplusplus
extern "C" {
#endif

// Creates an instance to the VAD structure.
VadInst *WebRtcVad_Create(void);

// Frees the dynamic memory of a specified VAD instance.
//
// - handle [i] : Pointer to VAD instance that should be freed.
void WebRtcVad_Free(VadInst *handle);

// Initializes a VAD instance.
//
// - handle [i/o] : Instance that should be initialized.
//
// returns        : 0 - (OK),
//                 -1 - (null pointer or Default mode could not be set).
int WebRtcVad_Init(VadInst *handle);

// Sets the VAD operating mode. A more aggressive (higher mode) VAD is more
// restrictive in reporting speech. Put in other words the probability of being
// speech when the VAD returns 1 is increased with increasing mode. As a
// consequence also the missed detection rate goes up.
//
// - handle [i/o] : VAD instance.
// - mode   [i]   : Aggressiveness mode (0, 1, 2, or 3).
//
// returns        : 0 - (OK),
//                 -1 - (null pointer, mode could not be set or the VAD instance
//                       has not been initialized).
int WebRtcVad_set_mode(VadInst *handle, int mode);

// Calculates a VAD decision for the |audio_frame|. For valid sampling rates
// frame lengths, see the description of WebRtcVad_ValidRatesAndFrameLengths().
//
// - handle       [i/o] : VAD Instance. Needs to be initialized by
//                        WebRtcVad_Init() before call.
// - fs           [i]   : Sampling frequency (Hz): 8000, 16000, or 32000
// - audio_frame  [i]   : Audio frame buffer.
// - frame_length [i]   : Length of audio frame buffer in number of samples.
// - keep_weight [i]   : return active voice weight
//
// returns              : 1 - (Active Voice),
//                        0 - (Non-active Voice),
//                       -1 - (Error)
int WebRtcVad_Process(VadInst *handle, int fs, const int16_t *audio_frame,
                      size_t frame_length, int keep_weight);

#ifdef __cplusplus
}
#endif

#endif  // COMMON_AUDIO_VAD_INCLUDE_WEBRTC_VAD_H_  // NOLINT
