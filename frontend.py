import streamlit as st
from loading import predict_and_view

st.title('😄 Emotion detection with deep learning')
buffer = st.file_uploader('upload image file here', type=['jpg', 'png', 'jpeg'])

if buffer is not None:
  st.write('loading')
  image, name, accuracy = predict_and_view(buffer, './effnet_model.pth')
  st.image(image, caption=f'prediction is {name} with accuracy {accuracy}')
