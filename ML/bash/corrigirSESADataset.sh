for file in /home/dimi/Downloads/SESA/train/*.wav; do 
	sox $file -c 1 -r 16000 -b 16 /home/dimi/Downloads/SESA_SOX/train/$(basename $file);
done

for file in /home/dimi/Downloads/SESA/test/*.wav; do 
	sox $file -c 1 -r 16000 -b 16 /home/dimi/Downloads/SESA_SOX/test/$(basename $file);
done