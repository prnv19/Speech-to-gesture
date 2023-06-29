import gradio as gr
import subprocess
import os

# Function to perform audio-to-video inference using shell script
def perform_inference(audio_path):
    current_directory = os.getcwd()
    try:
        os.remove(f"{current_directory}/result/Gesture_Video.mp4")  # Delete the file
    except OSError as e:
        pass
    # Replace `<shell_script_path>` with the actual path to your shell script
    subprocess.run('export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python', shell=True, check=True)
    shell_script_path = f"{current_directory}/generate.sh"
    subprocess.run(["sh", shell_script_path, audio_path])

    return

# Gradio interface
def audio_inference(audio_file):
    if audio_file is not None:
        perform_inference(audio_file)
        return "result/Gesture_Video.mp4"
    else:
        return "Please upload an audio file."
    


# Gradio interface setup

file_input = gr.inputs.Audio(label="Upload Audio", type="filepath")
iface = gr.Interface(fn=audio_inference, inputs=file_input, outputs="video", title="Audio to Gesture")
iface.launch()
