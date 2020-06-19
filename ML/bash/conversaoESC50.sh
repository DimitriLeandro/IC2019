# O ESC 50 SERA CONVERTIDO PARA 16KHZ 8BITS
# O VOLUME FOI AJUSTADO PARA 75% PARA EVITAR CLIPAGEM
for file in /home/dimi/Downloads/datasets/ESC_50_original/audio/*.wav; do 
	sox -v 0.75 $file -c 1 -r 16000 -b 8 /home/dimi/Downloads/datasets/ESC_50_16kHz_8bits/audio/$(basename $file);
done