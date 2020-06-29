# O ESC 50 SERA CONVERTIDO PARA 16KHZ 8BITS
# O VOLUME FOI AJUSTADO PARA 75% PARA EVITAR CLIPAGEM

for file in /home/dimi/Downloads/datasets/SESA_v2_16kHz_8bits/original_float/*.wav; do 
	sox -v 0.9 $file -c 1 -r 16000 -b 8 /home/dimi/Downloads/datasets/SESA_v2_16kHz_8bits/original/$(basename $file);
done

for file in /home/dimi/Downloads/datasets/SESA_v2_16kHz_8bits/silencio_float/*.wav; do 
	sox -v 0.9 $file -c 1 -r 16000 -b 8 /home/dimi/Downloads/datasets/SESA_v2_16kHz_8bits/silencio/$(basename $file);
done

for file in /home/dimi/Downloads/datasets/SESA_v2_16kHz_8bits/original_float/*.wav; do 
	sox -v 0.9 $file -c 1 -r 16000 -b 16 /home/dimi/Downloads/datasets/SESA_v2_16kHz_16bits/original/$(basename $file);
done

for file in /home/dimi/Downloads/datasets/SESA_v2_16kHz_8bits/silencio_float/*.wav; do 
	sox -v 0.9 $file -c 1 -r 16000 -b 16 /home/dimi/Downloads/datasets/SESA_v2_16kHz_16bits/silencio/$(basename $file);
done