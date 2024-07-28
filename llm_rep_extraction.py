from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoModel, AutoTokenizer
from pathlib import Path
import torchaudio
import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--audio_dir', type=str, help='Audio folder path', default="processed_dataset")
    parser.add_argument('--rep_dir', type=str, help='Path to save representation files', default="llm_representations")
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='.wav')
    parser.add_argument('--split_seed', type=int, help="Random seed", default=42)
    parser.add_argument('--valid_set_size', type=float, default=32)
    args, unknown = parser.parse_known_args()
    exts = args.exts.split(',')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = json.load(f)
    sample_rate = cfg.get('sample_rate')
    
    # Load speech-to-text model and tokenizer
    stt_model_path = cfg.get('stt_model_path')
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_path).eval().to(device)
    stt_tokenizer = Wav2Vec2Tokenizer.from_pretrained(stt_model_path)
    
    # Load the LLM and tokenizer
    llm_model_path = cfg.get('llm_model_path')
    llm_model = AutoModel.from_pretrained(llm_model_path).eval().to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    target_layer = cfg.get('llm_model_layer')
    
    path = Path(args.audio_dir)
    #file_list = [str(file) for file in path.glob("*.wav")]
    file_list = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]
    
    if args.valid_set_size != 0 and args.valid_set_size < 1:
        valid_set_size = int(len(file_list) * args.valid_set_size)
    else:
        valid_set_size = int(args.valid_set_size)
    train_file_list = f"llm_{cfg.get('train_files')}"
    valid_file_list = f"llm_{cfg.get('valid_files')}"
    segment_size = cfg.get('segment_size')
    random.seed(args.split_seed)
    random.shuffle(file_list)
    print(f'A total of {len(file_list)} samples will be processed, and {valid_set_size} of them will be included in the validation set.')
    
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav, sr = torchaudio.load(audio_file)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.size(-1) < segment_size:
                print("@@@@@@@@ wav.size(-1) < segment_size!! Check if dataset was processed correctly.")
                wav = torch.nn.functional.pad(wav, (0, segment_size - wav.size(-1)), 'constant')
            
            # Step 1: Convert audio to text
            input_values = stt_tokenizer(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
            logits = stt_model(input_values.to(stt_model.device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = stt_tokenizer.batch_decode(predicted_ids)[0]
            
            # Step 2: Pass text through LLM
            llm_inputs = llm_tokenizer(transcription, return_tensors="pt", truncation=True, 
                                       padding="max_length", max_length=logits.shape[1]).to(device) 
            llm_outputs = llm_model(**llm_inputs, output_hidden_states=True)
            # Without padding:
            # llm_inputs = llm_tokenizer(transcription, return_tensors="pt").to(device)
            # llm_outputs = llm_model(**llm_inputs, output_hidden_states=True)
            
            # Step 3: Extract LLM representation
            if target_layer == 'avg':
                rep = torch.mean(torch.stack(llm_outputs.hidden_states), axis=0)
            else:
                rep = llm_outputs.hidden_states[target_layer]
            
            # Step 4: Save the representation
            rep_file = audio_file.replace(args.audio_dir, args.rep_dir).split('.')[0] + '.llm.npy'
            rep_sub_dir = '/'.join(rep_file.split('/')[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            np.save(rep_file, rep.detach().cpu().numpy())
            #to save as file format
            if i == 0 or i == valid_set_size:
                # First open the file in write mode to clear its content
                with open(valid_file_list if i < valid_set_size else train_file_list, 'w') as f:
                    f.write(audio_file + "\t" + rep_file + "\n")
            else:
                with open(valid_file_list if i < valid_set_size else train_file_list, 'a+') as f:
                    f.write(audio_file + "\t" + rep_file + "\n")
                    
            if i == 0:
                # Check values 
                print('input_values: ******************* \\n shape={}, mean={}, min={}, max={}'.format(input_values.shape, input_values.float().mean(), input_values.float().min(), input_values.float().max()))
                print('logits: ******************* \\n shape={}, mean={}, min={}, max={}'.format(logits.shape, logits.mean(), logits.min(), logits.max()))
                print('predicted_ids: ******************* \\n shape={}, first 5={}, last 5={}'.format(predicted_ids.shape, predicted_ids[:5], predicted_ids[-5:]))
                print('transcription: ******************* \\n {}'.format(transcription))
                print('llm_inputs: ******************* \\n shape={}, mean={}, min={}, max={}'.format(llm_inputs.input_ids.shape, llm_inputs.input_ids.float().mean(), llm_inputs.input_ids.float().min(), llm_inputs.input_ids.float().max()))
                print('llm_outputs: ******************* \\n shape={}, mean={}, min={}, max={}'.format(llm_outputs.last_hidden_state.shape, llm_outputs.last_hidden_state.mean(), llm_outputs.last_hidden_state.min(), llm_outputs.last_hidden_state.max()))
                print('representations: ******************* \\n shape={}, mean={}, min={}, max={}'.format(rep.shape, rep.mean(), rep.min(), rep.max()))