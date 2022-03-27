
for i in range(len(audio_files)):
              if audio_files[i] not in dic:
                wav = preprocess_wav("/content/drive/MyDrive/path_to_wavs/"+audio_files[i].split("/")[-1])
                dic[audio_files[i]] = wav
              else:
                wav = dic[audio_files[i]]
              emb = embed_utterance(wav,self.SpeakerEncoder.to(DEVICE))
              ge2e_emb.append(torch.Tensor(emb))
            ge2e_emb = torch.stack(ge2e_emb, axis = 0)
            ge2e_emb = ge2e_emb.to(DEVICE)
            print(ge2e_emb.shape)