for file in audio/fold2/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold2/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold2/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold2/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold2/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold2/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold2/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold2/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold2/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold2/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold2/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold2/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold2/$(basename $file);
done

for file in audio/fold3/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold3/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold3/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold3/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold3/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold3/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold3/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold3/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold3/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold3/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold3/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold3/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold3/$(basename $file);
done

for file in audio/fold4/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold4/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold4/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold4/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold4/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold4/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold4/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold4/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold4/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold4/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold4/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold4/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold4/$(basename $file);
done

for file in audio/fold5/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold5/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold5/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold5/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold5/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold5/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold5/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold5/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold5/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold5/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold5/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold5/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold5/$(basename $file);
done

for file in audio/fold6/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold6/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold6/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold6/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold6/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold6/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold6/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold6/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold6/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold6/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold6/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold6/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold6/$(basename $file);
done

for file in audio/fold7/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold7/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold7/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold7/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold7/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold7/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold7/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold7/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold7/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold7/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold7/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold7/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold7/$(basename $file);
done

for file in audio/fold8/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold8/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold8/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold8/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold8/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold8/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold8/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold8/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold8/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold8/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold8/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold8/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold8/$(basename $file);
done

for file in audio/fold9/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold9/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold9/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold9/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold9/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold9/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold9/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold9/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold9/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold9/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold9/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold9/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold9/$(basename $file);
done

for file in audio/fold10/*.wav; do 
	sox $file -c 1 -r 48000 -b 8 conversoes/8bits/48k/fold10/$(basename $file);
	sox $file -c 1 -r 48000 -b 16 conversoes/16bits/48k/fold10/$(basename $file);

	sox $file -c 1 -r 40000 -b 8 conversoes/8bits/40k/fold10/$(basename $file);
	sox $file -c 1 -r 40000 -b 16 conversoes/16bits/40k/fold10/$(basename $file);

	sox $file -c 1 -r 32000 -b 8 conversoes/8bits/32k/fold10/$(basename $file);
	sox $file -c 1 -r 32000 -b 16 conversoes/16bits/32k/fold10/$(basename $file);

	sox $file -c 1 -r 24000 -b 8 conversoes/8bits/24k/fold10/$(basename $file);
	sox $file -c 1 -r 24000 -b 16 conversoes/16bits/24k/fold10/$(basename $file);

	sox $file -c 1 -r 16000 -b 8 conversoes/8bits/16k/fold10/$(basename $file);
	sox $file -c 1 -r 16000 -b 16 conversoes/16bits/16k/fold10/$(basename $file);

	sox $file -c 1 -r 8000 -b 8 conversoes/8bits/8k/fold10/$(basename $file);
	sox $file -c 1 -r 8000 -b 16 conversoes/16bits/8k/fold10/$(basename $file);
done