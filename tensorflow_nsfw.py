import tensorflow as tf
import numpy as np
import re
import pickle

with open('nsfw_classifier_tokenizer.pickle', 'rb') as f: tokenizer = pickle.load(f)
with open('nsfw_classifier.pickle', 'rb') as f: model = pickle.load(f)

# Define the vocabulary size and embedding dimensions
vocab_size = 10000
embedding_dim = 64

# Pad the prompt and negative prompt sequences
max_sequence_length = 50


def preprocess(text, isfirst = True):
    if isfirst:
        if type(text) == str: pass
        elif type(text) == list:
            output = []
            for i in text:
                output.append(preprocess(i))
            return(output)
            

    text = re.sub('<.*?>', '', text)
    text = re.sub('\(+', '(', text)
    text = re.sub('\)+', ')', text)
    matchs = re.findall('\(.*?\)', text)
    
    for _ in matchs:
        text = text.replace(_, preprocess(_[1:-1], isfirst=False) )

    text = text.replace('\n', ',').replace('|',',')

    if isfirst: 
        output = text.split(',')
        output = list(map(lambda x: x.strip(), output))
        output = [x for x in output if x != '']
        return ', '.join(output)
        # return output

    return text


def postprocess(prompts, negative_prompts, outputs, print_percentage = True):
    for idx, i in enumerate(prompts):
        print('*****************************************************************')
        if print_percentage:
            print(f"prompt: {i}\nnegative_prompt: {negative_prompts[idx]}\npredict: {outputs[idx][0]} --{outputs[idx][1]}%")
        else:
            print(f"prompt: {i}\nnegative_prompt: {negative_prompts[idx]}\npredict: {outputs[idx][0]}")

# Make predictions on new data
prompt = ["a landscape with trees and mountains in the background", 'nude, sexy, 1girl, nsfw']
negative_prompt = ["nsfw", 'worst quality']

x_new = tokenizer.texts_to_sequences( preprocess(prompt) )
z_new = tokenizer.texts_to_sequences( preprocess(negative_prompt) )
x_new = tf.keras.preprocessing.sequence.pad_sequences(x_new, maxlen=max_sequence_length)
z_new = tf.keras.preprocessing.sequence.pad_sequences(z_new, maxlen=max_sequence_length)
y_new = model.predict([x_new, z_new])
y_new = list(map(lambda x:("NSFW", float("{:.2f}".format(x[0]*100)) ) if x[0]>0.5 else ("SFW", float("{:.2f}".format(100-x[0]*100))), y_new))


print("Prediction:", y_new)
postprocess(prompt, negative_prompt, y_new, print_percentage=True)