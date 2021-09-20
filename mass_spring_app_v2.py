# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Sep, 2021
'''

import io

from derivative import dxdt
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.linalg import svd
import streamlit as st
from sklearn.linear_model import Lasso


np.set_printoptions(precision=2)

matplotlib.use('agg')


def main():
    apptitle = 'Mass-Spring-Damper-ODE'
    st.set_page_config(
        page_title=apptitle,
        page_icon=':eyeglasses:',
        # layout='wide'
    )
    st.title('Spring-mass-damping system analysis')

    # level 1 font
    st.markdown("""
        <style>
        .L1 {
            font-size:40px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # level 2 font
    st.markdown("""
        <style>
        .L2 {
            font-size:20px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    #########################Objectives#########################

    st.markdown('<p class="L1">Objectives</p>', unsafe_allow_html=True)

    str_1 = '''1. Illustrate the motion of the mass with time captured by camera.'''
    str_2 = '''2. Reduce the dimension of the camera data and identify the intrinsic dimension by principal component analysis (PCA).'''
    str_3 = '''3. Identify governing physical parameters for the spring mass system (spring constant and damping)'''

    st.markdown('<p class="L2">{}</p>'.format(str_1), unsafe_allow_html=True)
    st.markdown('<p class="L2">{}</p>'.format(str_2), unsafe_allow_html=True)
    st.markdown('<p class="L2">{}</p>'.format(str_3), unsafe_allow_html=True)

    #########################Experimental set#########################

    # Experimental set

    st.markdown('<p class="L1">Experimental set</p>', unsafe_allow_html=True)
    st.image('src/schematic.png')

    st.markdown('<p class="L2">Videos:</p>', unsafe_allow_html=True)
    str_1 = """[1. Camera 1](https://drive.google.com/file/d/1-BukVXmKl5G-hmR5v1dCUw7tEUy0MXAl/view?usp=sharing)"""
    st.markdown(str_1)
    str_2 = """[2. Camera 2](https://drive.google.com/file/d/1qTe8MC7yvCOlFx2JDDX6W5rvmHKfRCYI/view?usp=sharing)"""
    st.markdown(str_2)
    str_3 = """[3. Camera 3](https://drive.google.com/file/d/1NNwmqHz6wA6kToc6Ydqvs_1Mb7yZjvZX/view?usp=sharing)"""
    st.markdown(str_3)

    #########################Load data#########################
    st.markdown('<p class="L1">Upload three camera dataset</p>',
            unsafe_allow_html=True)

    flag = ['New dataset', 'Default dataset']
    st.markdown('<p class="L2">Chosse a new dataset or use default dataset:</p>',
                unsafe_allow_html=True)
    use_new_data = st.selectbox('', flag, 1)

    # load dataset
    if use_new_data == 'New dataset':
        uploaded_file = st.file_uploader(
            'Choose a CSV file', accept_multiple_files=False)

    # button_dataset = st.button('Click once you have selected a dataset')
    # if button_dataset:
    # load dataset
    if use_new_data == 'New dataset':
        data = io.BytesIO(uploaded_file.getbuffer())
        df = pd.read_csv(data)
        list = np.loadtxt(open("src/21data-new.csv", "rb"),
                          delimiter=",", skiprows=1)
    elif use_new_data == 'Default dataset':
        file_path = 'src/21data-new.csv'
        df = pd.read_csv(file_path)
        list = np.loadtxt(open(file_path, "rb"),
                          delimiter=",", skiprows=1)

    lines = ['Camera 1', 'Camera 2', 'Camera 3']
    st.markdown('<p class="L2">Select a camera to show data:</p>',
                unsafe_allow_html=True)
    chosen_line = st.selectbox('', lines, 0)
    chosen_line = chosen_line.replace('Camera ', '')
    global ratio
    st.markdown('<p class="L2">Select a ratio to show data:</p>',
                unsafe_allow_html=True)
    ratio = st.slider('', 0.01, 1.0, 1.0)

    col1, col2, col3 = st.beta_columns(3)
    show_length = int(df.shape[0] * ratio)-1
    with col1:
        fig = plt.figure()
        plt.plot(df['t'][:show_length], df[f'x{chosen_line}'][:show_length])
        plt.xlabel('t', fontsize=30)
        plt.ylabel(f'x', fontsize=30)
        plt.title(f'x (Camera {chosen_line})', fontsize=34)
        plt.tick_params(labelsize=26)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = plt.figure()
        plt.plot(df['t'][:show_length], df[f'y{chosen_line}'][:show_length])
        plt.xlabel('t', fontsize=30)
        plt.ylabel(f'y', fontsize=30)
        plt.title(f'y (Camera {chosen_line})', fontsize=34)
        plt.tick_params(labelsize=26)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with col3:
        fig = plt.figure()
        plt.scatter(df[f'x{chosen_line}'][:show_length],
                    df[f'y{chosen_line}'][:show_length])
        plt.xlabel(f'x', fontsize=30)
        plt.ylabel(f'y', fontsize=30)
        plt.title(f'x-y (Camera {chosen_line})', fontsize=34)
        plt.tick_params(labelsize=26)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


    #three cameras dataset
    # list = np.loadtxt(open("src/21data-new.csv", "rb"), delimiter=",", skiprows=1)
    X = list[:, 1:7]
    t = list[:, 0]
    # print(X)
    nPoints = X.shape[0]
    # print(nPoints)

    Xavg = np.mean(X, axis=0)
    # print(Xavg)
    B = X - Xavg
    # print(B)
    plt.plot(t, B[:, 1], '-')
    plt.xlim(xmin=0, xmax=10)
    #plt.ylim(ymin = -60, ymax = 60)
    # plt.show()

    #print(B/math.sqrt(nPoints))
    # SVD
    U, s, VT = svd(B, full_matrices=False)
    # print(U)
    # print(np.diag(s))
    # print(VT)

    BV = B.dot(VT.T)
    C_BV = 1/(nPoints-1)*BV.T.dot(BV)
    # print(C_BV)

    st.markdown('<p class="L1">Dimension reduction by principal component analysis</p>',
                unsafe_allow_html=True)
    #PCA
    C_B = 1/(nPoints-1)*B.T.dot(B)
    # print(C_B)
    eigenvalues, eigenvectors = LA.eig(C_B)
    # print(eigenvalues, eigenvectors)
    # print(s*s/(nPoints-1))

    eigenvalues_str = ', '.join([str(round(i, 2))
                                 for i in eigenvalues.tolist()])
    # st.markdown('<p class="L2">Eigenvalues:</p>', unsafe_allow_html=True)
    # st.markdown(f'[{eigenvalues_str}]')

    # plot the log of eigenvalues
    x_major_locator = MultipleLocator(1)  # used to set x-axis tick interval
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # ax1.semilogy(range(1, 7), s, '-o', color='k')
    ax1.plot(range(1, 7), eigenvalues, '-o', color='k')
    ax1.xaxis.set_major_locator(x_major_locator)  # set x-axis tick interval
    ax1.set_xlabel('Principal Component Number', fontsize=16)
    ax1.set_ylabel('Eigenvalues', fontsize=16)
    plt.tight_layout()
    plt.grid()
    st.pyplot(fig, clear_figure=True)
    st.markdown('## This system is one-dimensional.', unsafe_allow_html=True)

    ##########################ODE##########################
    st.markdown('<p class="L1">Recover ordinary differential equation (ODE)</p>',
                unsafe_allow_html=True)

    st.markdown('<p class="L2">Reduce 3 set of camera data into 1 set of data.</p>',
                unsafe_allow_html=True)
    #recover ODE
    z = B[:, 1]
    Z = np.stack((z), axis=-1)  # First column is x, second is y
    #print(X)
    #plot x
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, z, '-', color='blue')
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.set_ylabel("Z", fontsize=20)
    ax.tick_params(labelsize=20)
    # ax.legend(fontsize=20)
    #ax.set_xlim(-0.05, 1.05)
    #ax.set_ylim(-6.5, 2.5)
    st.pyplot(fig, clear_figure=True)


    st.markdown('<p class="L2">Calculate the first-order and second-order derivatives</p>',
            unsafe_allow_html=True)
    z_dot = dxdt(z, t, kind="finite_difference", k=1)
    #z_dot=dxdt(z, t, kind="trend_filtered", order=3, alpha=0.1)
    z_2dot = dxdt(z_dot, t, kind="finite_difference", k=1)

    Z_2dot = np.stack((z_2dot), axis=-1)  # First column is x, second is y

    col1, col2 = st.beta_columns(2)
    with col1:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(t, z_dot, '-', color='blue')
        ax.set_xlabel("Time (s)", fontsize=30)
        ax.tick_params(labelsize=30)
        ax.set_ylabel(r"$\dot{z} $", fontsize=30)
        #ax.set_xlim(-0.05, 1.05)
        #ax.set_ylim(-6.5, 2.5)
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(t, z_2dot, '-', color='blue')
        ax.set_xlabel("Time (s)", fontsize=30)
        ax.set_ylabel(r"$\ddot{z} $", fontsize=30)
        ax.tick_params(labelsize=30)
        #ax.set_xlim(-0.05, 1.05)
        #ax.set_ylim(-6.5, 2.5)
        st.pyplot(fig, clear_figure=True)

    # st.markdown('<p class="L2">Sparse coefficients fitting to recover ODE</p>',
                # unsafe_allow_html=True)

    # st.markdown('$\ddot{z} = c_1 * z + c_2 * \dot{z}$')

    st.image('src/sindy.jpg')

    theta1 = z
    theta2 = z_dot

    THETA = np.stack((theta1, theta2), axis=-1)  # First column is x, second is y
    # print(THETA)

    model = Lasso(alpha=1e-3, max_iter=200, fit_intercept=False)

    model.fit(THETA, Z_2dot)

    r_sq = model.score(THETA, Z_2dot)

    st.markdown('<p class="L2">Identified coefficients:</p>', unsafe_allow_html=True)
    st.markdown('$\\alpha: {}$'.format(round(model.coef_[0], 4)))
    st.markdown('$\\beta: {}$'.format(round(model.coef_[1], 4)))

    st.markdown('<p class="L2">Fitting performance: The coefficient of determination is {}</p>'.format(
    round(r_sq, 2)), unsafe_allow_html=True)
    
    st.markdown('<p class="L2">Set the mass:</p>', unsafe_allow_html=True)
    m = float(st.text_input('', 0.1))
    k = -model.coef_[0]*m
    c = -model.coef_[1]*m

    st.markdown('<p class="L2">Recovered spring constant k: {} N/m</p>'.format(round(k, 4)), unsafe_allow_html=True)
    st.markdown('<p class="L2">Recovered damping coefficient c: {} Ns/m</p>'.format(
        round(c, 4)), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
