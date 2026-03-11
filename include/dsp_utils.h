/*
 * dsp_utils.h - Primitivas de Processamento Digital de Sinais
 *
 * Pre-enfase, janelamento Hamming, autocorrelacao, FFT radix-2
 * e funcoes auxiliares para analise de audio.
 */

#ifndef DSP_UTILS_H
#define DSP_UTILS_H

/*
 * Aplica filtro de pre-enfase: y[n] = x[n] - alpha * x[n-1]
 * Modifica o array in-place. Primeiro elemento fica inalterado.
 */
void dsp_pre_emphasis(float *signal, int n, float alpha);

/*
 * Aplica janela de Hamming in-place.
 * w[i] = 0.54 - 0.46 * cos(2*pi*i / (n-1))
 */
void dsp_hamming_window(float *frame, int n);

/*
 * Calcula autocorrelacao R[k] para k = 0..max_lag-1
 * R[k] = sum_{i=0}^{n-k-1} signal[i] * signal[i+k]
 * O array out deve ter pelo menos max_lag elementos.
 */
void dsp_autocorrelation(const float *signal, int n, float *out, int max_lag);

/*
 * FFT radix-2 in-place (Cooley-Tukey).
 * real[] e imag[] devem ter tamanho n, onde n eh potencia de 2.
 * Se o sinal tem menos que n amostras, preencher com zeros.
 */
void dsp_fft(float *real, float *imag, int n);

/*
 * Calcula o espectro de magnitude |X[k]| = sqrt(re^2 + im^2)
 * Resultado em mag[], que deve ter pelo menos n/2 + 1 elementos
 * (apenas a metade positiva do espectro).
 */
void dsp_magnitude_spectrum(const float *real, const float *imag, int n, float *mag);

/*
 * Calcula o espectro de potencia |X[k]|^2 / n
 * Resultado em power[], que deve ter pelo menos n/2 + 1 elementos.
 */
void dsp_power_spectrum(const float *real, const float *imag, int n, float *power);

/*
 * Retorna a menor potencia de 2 >= n.
 */
int dsp_next_power_of_2(int n);

/*
 * Calcula a energia total do sinal: sum(x[i]^2) / n
 */
float dsp_energy(const float *signal, int n);

/*
 * US-010: Wavelet Denoising via Haar DWT + soft thresholding.
 * Aplica denoising in-place usando o limiar universal de Donoho & Johnstone:
 *   sigma = RMS(coef_detalhe_nivel1), T = sigma * sqrt(2 * log(n))
 * levels: numero de niveis de decomposicao (recomendado: 3-5).
 * Apenas a porcao com potencia de 2 <= n e processada.
 */
void dsp_wavelet_denoise(float *signal, int n, int levels);

#endif /* DSP_UTILS_H */
