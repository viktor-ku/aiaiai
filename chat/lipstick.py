from ai.text2image.flux import make_pipe, snap

pipe = make_pipe()

image = snap(pipe, prompt="")

image.save(f"output/")
