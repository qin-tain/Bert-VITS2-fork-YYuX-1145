import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import webbrowser
import time
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence,_symbol_to_id, get_bert
from text.cleaner import clean_text
from scipy.io import wavfile


device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "简体中文": "[ZH]",
}
lang = ['简体中文']
def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    print([f"{p}{t}" for p, t in zip(phone, tone)])
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language
'''
def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn
'''
dev='cuda'
def infer(text, sdp_ratio, noise_scale, noise_scale_w,length_scale,sid):
    bert, phones, tones, lang_ids = get_text(text,"ZH", hps,)
    print(sid)
    with torch.no_grad():
        x_tst=phones.to(dev).unsqueeze(0)
        tones=tones.to(dev).unsqueeze(0)
        lang_ids=lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(dev)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids,bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        #del stn_tst,tones,lang_ids,bert, x_tst, x_tst_lengths, sid
        return "Success",(hps.data.sampling_rate, audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./configs\config.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)


    net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(dev)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None,skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(hps.data.spk2id.keys())
    #inf = infer(net_g, hps, speaker_ids)
    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="你知道吗，我最喜欢的季节是冬天。冬天的雪花纷飞，给人一种宁静而美丽的感觉。我喜欢在雪地里漫步，感受雪花轻轻落在脸上的触感。冬天的夜晚，星星点点的繁星在寒冷的天空中闪烁，让人感到无比宁静和神秘。", elem_id=f"tts-input")
                    # select character
                    char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
                    language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
                    sdp_ratio = gr.Slider(minimum=0.1, maximum=1, value=0.2, step=0.1,
                                                label='SDP/DP混合比')
                    noise_scale = gr.Slider(minimum=0.1, maximum=1, value=0.5, step=0.1,
                                                label='noise')
                    noise_scale_w = gr.Slider(minimum=0.1, maximum=1, value=0.9, step=0.1,
                                                label='noisew')
                    length_scale = gr.Slider(minimum=0.1, maximum=2, value=1.0, step=0.1,
                                                label='length')
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(infer,
                              inputs=[textbox,sdp_ratio,noise_scale,noise_scale_w,length_scale,char_dropdown],
                              outputs=[text_output, audio_output])
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)

