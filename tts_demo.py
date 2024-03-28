from tts_module import do_tts

# audio_output_fullpath = f"{working_dir}/{audio_output_filename}"

# make an agent that is responsible for giving updates throughout 
# code and relaying them back to me; e.g. "Now we're going to pass that off to the X agent and do XYZ.."
# it needs to review and summarize the actions taking place and summarize them into a single update to provide to dev

tts_text = "I'm sorry Dave, I can't do that."

do_tts(tts_text, "tts-1", "onyx", 0.95)
