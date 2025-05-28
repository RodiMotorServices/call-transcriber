#!/usr/bin/env python3
"""
Advanced Audio Enhancement for Call Transcriber
Provides sophisticated audio preprocessing for improved transcription quality
"""

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class AudioEnhancer:
    """Advanced audio enhancement for speech recognition"""
    
    def __init__(self):
        self.target_sr = 16000
        self.target_channels = 1
    
    def spectral_subtraction_noise_reduction(self, audio: np.ndarray, sr: int, 
                                           noise_factor: float = 1.0, 
                                           alpha: float = 2.0) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction
        Estimates noise from first 0.5 seconds and subtracts it from the entire signal
        """
        # Convert to frequency domain
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from the first 0.5 seconds
        noise_frames = int(0.5 * sr / 512)  # 0.5 seconds worth of frames
        noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_factor * noise_magnitude
        
        # Ensure we don't go below a certain threshold (0.1 * original magnitude)
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # Reconstruct the signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
        
        return enhanced_audio
    
    def voice_activity_detection(self, audio: np.ndarray, sr: int, 
                                frame_length: int = 2048, 
                                hop_length: int = 512) -> np.ndarray:
        """
        Detect voice activity and suppress non-speech segments
        """
        # Compute spectral centroid and zero crossing rate
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                              hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, 
                                               hop_length=hop_length)[0]
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                 hop_length=hop_length)[0]
        
        # Normalize features
        spectral_centroids_norm = (spectral_centroids - np.mean(spectral_centroids)) / np.std(spectral_centroids)
        zcr_norm = (zcr - np.mean(zcr)) / np.std(zcr)
        rms_norm = (rms - np.mean(rms)) / np.std(rms)
        
        # Simple VAD: voice activity when energy is high and spectral centroid is reasonable
        vad = (rms_norm > -1.0) & (spectral_centroids_norm > -2.0) & (zcr_norm < 2.0)
        
        # Smooth the VAD decisions (median filter)
        vad_smooth = signal.medfilt(vad.astype(float), kernel_size=5)
        
        return vad_smooth > 0.5
    
    def apply_telephony_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply telephony frequency filtering (300-3400 Hz)
        Simulates phone line frequency response
        """
        # Design bandpass filter
        nyquist = sr / 2
        low_freq = 300 / nyquist
        high_freq = 3400 / nyquist
        
        # Design butterworth bandpass filter
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def dynamic_range_compression(self, audio: np.ndarray, 
                                threshold: float = 0.1, 
                                ratio: float = 4.0,
                                attack_time: float = 0.003,
                                release_time: float = 0.1,
                                sr: int = 16000) -> np.ndarray:
        """
        Apply dynamic range compression to normalize volume levels
        """
        # Convert times to samples
        attack_samples = int(attack_time * sr)
        release_samples = int(release_time * sr)
        
        # Initialize envelope follower
        envelope = np.zeros_like(audio)
        
        for i in range(1, len(audio)):
            # Get current sample magnitude
            current_mag = abs(audio[i])
            
            # Attack or release
            if current_mag > envelope[i-1]:
                # Attack - fast rise
                alpha = 1.0 - np.exp(-1.0 / attack_samples)
            else:
                # Release - slow fall
                alpha = 1.0 - np.exp(-1.0 / release_samples)
            
            envelope[i] = alpha * current_mag + (1 - alpha) * envelope[i-1]
        
        # Apply compression
        compressed = np.copy(audio)
        mask = envelope > threshold
        
        # Compress signals above threshold
        gain_reduction = threshold + (envelope[mask] - threshold) / ratio
        compressed[mask] = audio[mask] * (gain_reduction / envelope[mask])
        
        return compressed
    
    def enhance_speech_clarity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance speech clarity using formant emphasis
        """
        # Apply pre-emphasis filter (high-pass)
        pre_emphasis = 0.97
        emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Enhance formant frequencies (speech resonances around 500-2000 Hz)
        # Design a gentle boost filter
        nyquist = sr / 2
        center_freq = 1250 / nyquist  # Center around 1250 Hz
        
        # Design a peaking filter for formant enhancement
        b, a = signal.iirpeak(center_freq, Q=1.0)
        enhanced = signal.filtfilt(b, a, emphasized)
        
        return enhanced
    
    def process_audio_file(self, input_path: str, 
                          enhance_speech: bool = True,
                          reduce_noise: bool = True,
                          apply_vad: bool = False,
                          telephony_filter: bool = True,
                          normalize_volume: bool = True) -> str:
        """
        Process an audio file with comprehensive enhancements
        
        Args:
            input_path: Path to input audio file
            enhance_speech: Apply speech clarity enhancement
            reduce_noise: Apply noise reduction
            apply_vad: Apply voice activity detection
            telephony_filter: Apply telephony frequency filtering
            normalize_volume: Normalize volume levels
            
        Returns:
            Path to processed audio file
        """
        # Load audio with librosa
        audio, original_sr = librosa.load(input_path, sr=None, mono=True)
        
        # Resample to target sample rate if needed
        if original_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        
        print(f"Loaded audio: {len(audio)/self.target_sr:.1f}s at {self.target_sr}Hz")
        
        # Apply enhancements in order
        if normalize_volume:
            # Initial normalization
            audio = audio / np.max(np.abs(audio))
            audio = audio * 0.8  # Leave some headroom
        
        if reduce_noise:
            print("Applying noise reduction...")
            audio = self.spectral_subtraction_noise_reduction(audio, self.target_sr)
        
        if telephony_filter:
            print("Applying telephony filter...")
            audio = self.apply_telephony_filter(audio, self.target_sr)
        
        if enhance_speech:
            print("Enhancing speech clarity...")
            audio = self.enhance_speech_clarity(audio, self.target_sr)
        
        # Apply dynamic range compression
        print("Applying dynamic range compression...")
        audio = self.dynamic_range_compression(audio, sr=self.target_sr)
        
        if apply_vad:
            print("Applying voice activity detection...")
            vad = self.voice_activity_detection(audio, self.target_sr)
            
            # Expand VAD decisions to audio samples
            hop_length = 512
            vad_samples = np.repeat(vad, hop_length)[:len(audio)]
            
            # Gradually fade out non-speech segments instead of hard gating
            fade_samples = int(0.01 * self.target_sr)  # 10ms fade
            for i in range(len(vad_samples)):
                if not vad_samples[i]:
                    # Gradual fade
                    fade_factor = 0.1  # Keep 10% of non-speech
                    audio[i] *= fade_factor
        
        # Final normalization
        if normalize_volume:
            audio = audio / np.max(np.abs(audio))
            audio = audio * 0.9  # Leave headroom
        
        # Save to temporary file
        temp_wav = tempfile.mktemp(suffix=".wav")
        sf.write(temp_wav, audio, self.target_sr)
        
        print(f"Enhanced audio saved: {len(audio)/self.target_sr:.1f}s")
        return temp_wav
    
    def analyze_audio_quality(self, audio_path: str) -> dict:
        """
        Analyze audio quality metrics
        """
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Basic metrics
        duration = len(audio) / sr
        rms_energy = np.sqrt(np.mean(audio**2))
        peak_amplitude = np.max(np.abs(audio))
        
        # Spectral metrics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Signal-to-noise ratio estimation (rough)
        # Assume first and last 0.5 seconds are mostly noise
        noise_duration = min(0.5, duration / 4)
        noise_samples = int(noise_duration * sr)
        
        if len(audio) > 2 * noise_samples:
            noise_start = audio[:noise_samples]
            noise_end = audio[-noise_samples:]
            noise_power = np.mean(np.concatenate([noise_start, noise_end])**2)
            signal_power = np.mean(audio**2)
            snr_estimate = 10 * np.log10(signal_power / max(noise_power, 1e-10))
        else:
            snr_estimate = None
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'rms_energy': rms_energy,
            'peak_amplitude': peak_amplitude,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossing_rate': zero_crossing_rate,
            'estimated_snr': snr_estimate,
            'dynamic_range': 20 * np.log10(peak_amplitude / max(rms_energy, 1e-10))
        }

def main():
    """CLI for audio enhancement"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Audio Preprocessing for Speech Recognition")
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output", "-o", help="Output file (default: enhanced_<input>)")
    parser.add_argument("--no-speech-enhance", action="store_true", help="Disable speech enhancement")
    parser.add_argument("--no-noise-reduction", action="store_true", help="Disable noise reduction")
    parser.add_argument("--enable-vad", action="store_true", help="Enable voice activity detection")
    parser.add_argument("--no-telephony-filter", action="store_true", help="Disable telephony filter")
    parser.add_argument("--no-normalize", action="store_true", help="Disable volume normalization")
    parser.add_argument("--analyze", action="store_true", help="Analyze audio quality")
    
    args = parser.parse_args()
    
    enhancer = AudioEnhancer()
    
    if args.analyze:
        quality = enhancer.analyze_audio_quality(args.input_file)
        print("\nAudio Quality Analysis:")
        print(f"Duration: {quality['duration']:.1f}s")
        print(f"Sample Rate: {quality['sample_rate']}Hz")
        print(f"RMS Energy: {quality['rms_energy']:.4f}")
        print(f"Peak Amplitude: {quality['peak_amplitude']:.4f}")
        print(f"Spectral Centroid: {quality['spectral_centroid']:.1f}Hz")
        print(f"Spectral Bandwidth: {quality['spectral_bandwidth']:.1f}Hz")
        print(f"Zero Crossing Rate: {quality['zero_crossing_rate']:.4f}")
        if quality['estimated_snr']:
            print(f"Estimated SNR: {quality['estimated_snr']:.1f}dB")
        print(f"Dynamic Range: {quality['dynamic_range']:.1f}dB")
    
    if not args.output:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}_enhanced.wav"
    
    enhanced_file = enhancer.process_audio_file(
        args.input_file,
        enhance_speech=not args.no_speech_enhance,
        reduce_noise=not args.no_noise_reduction,
        apply_vad=args.enable_vad,
        telephony_filter=not args.no_telephony_filter,
        normalize_volume=not args.no_normalize
    )
    
    # Copy to final output location
    import shutil
    shutil.move(enhanced_file, args.output)
    print(f"Enhanced audio saved to: {args.output}")

if __name__ == "__main__":
    main() 