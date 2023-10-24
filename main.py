import subprocess
import json
import argparse


def run_vad_and_asr(audiofile_path):
    audiofile_path = {"audio_filepath": audiofile_path}
    input_manifest="input_manifest.json"
    with open(input_manifest, "w") as json_file:

        json.dump(audiofile_path, json_file)

    vad_out_manifest_filepath="vad_out.json"



    vad_model = "vad_multilingual_marblenet"

    segmented_output_manifest="asr_segmented_output_manifest.json"
    asr_model="stt_en_citrinet_1024_gamma_0_25"

    command_vad = (
        f"python3 scripts/vad_infer.py "
        f"--config-path='../conf/vad' "
        f"--config-name='vad_inference_postprocessing.yaml' "
        f"dataset={input_manifest} "
        f"vad.model_path={vad_model} "
        f"frame_out_dir='frame_out' "
        f"vad.parameters.window_length_in_sec=0.63 "
        f"vad.parameters.postprocessing.onset=0.7 "
        f"vad.parameters.postprocessing.offset=0.4 "
        f"vad.parameters.postprocessing.min_duration_on=1 "
        f"vad.parameters.postprocessing.min_duration_off=0.5 "
        f"out_manifest_filepath={vad_out_manifest_filepath}"
    )


    command_asr = (
        f"python3 scripts/transcribe_speech.py "
        f"pretrained_name={asr_model} "
        f"dataset_manifest={vad_out_manifest_filepath} "
        f"batch_size=32 "
        f"amp=True "
        f"output_filename={segmented_output_manifest}"
    )

    subprocess.call(command_vad, shell=True)
    subprocess.call(command_asr, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка аудиофайла с использованием VAD и ASR")
    parser.add_argument("audiofile_path", type=str, help="Путь к аудиофайлу")
    args = parser.parse_args()
    results=run_vad_and_asr(args.audiofile_path)
    with open("asr_segmented_output_manifest.json", "r") as json_file:
        for line in json_file:
            data = json.loads(line)
            start_phrase = data.get("offset")
            end_phrase = start_phrase + data.get("duration")
            pred_text = data.get("pred_text")
            print(f"Начало фразы: {start_phrase}, Конец фразы: {end_phrase}, Текст: {pred_text}")
