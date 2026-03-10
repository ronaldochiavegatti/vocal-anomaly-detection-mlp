#ifndef WAV_AUGMENT_H
#define WAV_AUGMENT_H

/*
 * wav_augment.h - Augmentacao no dominio do audio (float normalizado [-1,1])
 *
 * Tecnicas: ruido branco, ganho, pitch shift, time stretch.
 * Todas as funcoes operam sobre amostras float normalizadas (mesmo formato de WavFile).
 * Funcoes in-place: modifica samples[] sem alterar n.
 */

/*
 * Adiciona ruido branco Gaussiano na SNR especificada (dB).
 * snr_db tipico: 20-30 dB.
 */
void wav_aug_noise(float *samples, int n, float snr_db);

/*
 * Aplica variacao de ganho (amplitude scaling).
 * db > 0 aumenta, db < 0 diminui. Resultado clamped em [-1, 1].
 */
void wav_aug_gain(float *samples, int n, float db);

/*
 * Time stretch in-place: reamostrar por 1/factor e ajustar a n amostras.
 * factor > 1  → mais lento (dilatacao temporal)
 * factor < 1  → mais rapido (compressao temporal)
 */
void wav_aug_stretch(float *samples, int n, float factor);

/*
 * Pitch shift in-place: deslocar F0 por semitones semitons (±1-2).
 * Implementado como reamostragem por 2^(semitones/12), ajustado a n amostras.
 * Semitones > 0 → pitch mais alto; semitones < 0 → pitch mais baixo.
 */
void wav_aug_pitch(float *samples, int n, float semitones);

#endif /* WAV_AUGMENT_H */
