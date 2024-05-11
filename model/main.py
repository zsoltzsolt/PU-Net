from open3d import *
import pickle
import numpy as np
import torch
import streamlit as st
import os

def plot_point_cloud(xml_file_path):
    with open('saved_models/E9_85.pkl', 'rb') as f:
        x = pickle.load(f)

    data = np.loadtxt(xml_file_path)[:, :3]

    output = x(torch.tensor([data, data, data, data], dtype=torch.float32, device='cuda'))
    output_list = output[0].cpu().detach().numpy().tolist()

    temp_output_file = 'temp_output.xyz'
    with open(temp_output_file, 'w') as f:
        for i in output_list:
            for j in i:
                f.write(str(j) + ' ')
            f.write('\n')

    cloud = io.read_point_cloud(temp_output_file)

    st.write("Original Point Cloud:")
    st.write(cloud)
    visualization.draw_geometries([cloud])
def main():
    st.title('Point Cloud Visualization')
    uploaded_file = st.file_uploader("Choose an XYZ file", type=['xyz'])

    if uploaded_file is not None:
        with open(f"uploads/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getvalue())

    plot_point_cloud(os.path.join("uploads", uploaded_file.name))


if __name__ == "__main__":
    main()