import streamlit as st
import shakespearee as ss
st.title("Next Character Prediction")

seed_text = st.text_input("Enter seed text with only lower case letters:", "To be or not to be")
next_chars = st.slider("Number of characters to generate:", 1, 500, 100)

if st.button("Generate"):

    generated_text = ss.generate_name(ss.model, seed_text, ss.itos, ss.stoi, ss.block_size, next_chars)
    # generate_name(model, inp, itos, stoi, block_size, max_len=no_of_chars)
    st.write("Generated Text:")
    st.write(generated_text)

# ----------------------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import torch
# from torch import nn
# import pandas as pd
# import numpy as np

# # Load the pre-trained model and vocabulary
# # model = torch.load('shakespeare_model.pth')
# # itos = np.load('itos.npy')

# # Streamlit UI
# st.title('Shakespearean Text Generator')

# # Input fields

# embedding_size = st.number_input('Enter Embedding Size', min_value=1, max_value=100, value=4, step=1)
# num_characters = st.number_input('Enter Number of Characters to Generate', min_value=1, max_value=100, value=10, step=1)
# seed_text = st.text_input('Enter Seed Text', value='To be or not to be, that is the question.')

# # Function to generate text
# def generate_text(model, seed_text, num_characters, embedding_size):
#     model.eval()
#     seed_text = seed_text.lower()
#     context = [0] * (ss.block_size - len(seed_text))
#     for char in seed_text:
#         ix = ss.stoi[char] if char in ss.stoi else 0
#         context.append(ix)
#     result = seed_text
#     with torch.no_grad():
#         for _ in range(num_characters):
#             x = torch.tensor(context).view(1, -1).to(ss.device)
#             y_pred = model(x)
#             ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
#             char = ss.itos[ix]
#             result += char
#             context = context[1:] + [ix]
#     return result

# # Generate and display text
# if st.button('Generate Text'):
#     generated_text = generate_text(ss.model, seed_text, num_characters, embedding_size)
#     st.write('Generated Text:', generated_text)

# --------------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import shakespeare as ss

# # Function to generate next k characters
# def generate_next_characters(embedding_size, num_chars, seed_text):
#     block_size = 5  # Fixed block size used in shakespeare.py
#     model = ss.NextChar(block_size, len(ss.stoi), embedding_size, 10).to(ss.device)
#     generated_text = ss.generate_name(model, ss.itos, ss.stoi, block_size, max_len=num_chars, seed_text=seed_text)
#     return generated_text

# # Main Streamlit app
# def main():
#     st.title("Shakespearean Text Generator")

#     # User input for embedding size, number of characters, and seed text
#     embedding_size = st.slider("Embedding Size:", min_value=1, max_value=10, value=4, step=1)
#     num_chars = st.slider("Number of Characters to Generate:", min_value=1, max_value=100, value=10, step=1)
#     seed_text = st.text_input("Seed Text:", "")

#     # Button to generate next characters
#     if st.button("Generate"):
#         if seed_text:
#             generated_text = generate_next_characters(embedding_size, num_chars, seed_text)
#             st.success(f"Generated Text: '{generated_text}'")
#         else:
#             st.error("Please provide a seed text.")

# if __name__ == "__main__":
#     main()
