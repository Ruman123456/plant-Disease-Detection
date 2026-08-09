import os

CHUNK_SIZE = 50 * 1024 * 1024
f = open('model.tflite', 'rb')
i = 0
while True:
    chunk = f.read(CHUNK_SIZE)
    if not chunk:
        break
    with open(f'model.tflite.part{i}', 'wb') as out:
        out.write(chunk)
    i += 1
f.close()
print("Split done")
