# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:05:04 2025

@author: ivyzh
"""

# install required libraries
import streamlit as st
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import scipy.stats as stats

# title of app
st.title('Histogram Generator')

# appearance: tabs, columns, expanders, etc 
tab1, tab2 = st.tabs(["Data", "Histogram"]) #spilt up webpage into two pages: one for data input and one to display the histogram

with tab1: 
    col1, col2 = st.columns(2)
    with col1: 
        st.write("Manually input option") # user manually input
        user_input = st.text_input("Enter numbers separated by commas")
        if user_input is not None:
            df = pd.DataFrame(columns=["values"])
            try:
                numbers = [float(x.strip()) for x in user_input.split(",")]
                df = pd.DataFrame({"values": numbers})
            
            except ValueError:
                st.error("Please enter valid numbers separated by commas.")
        
            values = df["values"]
            
    with col2:
        st.write("File upload option") # user CSV file upload
       
        uploaded_file = st.file_uploader("Upload a CSV file", type = ["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)  # store data
            
            st.write("Data Preview")
            st.dataframe(df)
            
            select_col = st.selectbox(
                "Select a numeric column", 
                df.select_dtypes(include="number").columns
            )
            values = df[select_col].dropna().values  
            df["values"] = values                
                                      
with tab2:
    col1, col2 = st.columns(2)
   
    if user_input:  # only run if user typed something
        numbers = [float(x.strip()) for x in user_input.split(",")]
        df = pd.DataFrame({"values": numbers})
    
    with col1:
        choice = st.text_input("Enter distribution type:")
        st.write("Options: Gamma, Normal, Uniform, Left-skewed, Right-skewed, Exponential, Beta, Pareto, Lognormal, Weibull_min, Cauchy, Chi2")    
        st.subheader("Manual Fitting")
    with col2: 
        # make a histogram   
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.hist(df["values"], bins=20, density=True)
        axis.set_title("Histogram")   
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")
       
        from scipy.stats import gamma
        from scipy.stats import norm
        from scipy.stats import uniform
        from scipy.stats import gumbel_r
        from scipy.stats import gumbel_l
        from scipy.stats import expon
        from scipy.stats import beta
        from scipy.stats import pareto
        from scipy.stats import lognorm
        from scipy.stats import weibull_min
        from scipy.stats import cauchy
        from scipy.stats import chi2
        
        # creates the output area
        counts, bins = np.histogram(values, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # layer the distribution curve on top of the histogram
        if choice == "Gamma":
            params = gamma.fit(df["values"])
            dist = gamma(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)    
           
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    beta1 = st.slider("Beta(scale)", 0.1, 10.0, 1.0)
                    fitted_dist = stats.gamma(a=alpha, loc=loc, scale=beta1)
            
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
                
                else: 
                    shape, loc, scale = gamma.fit(values)
                    mu = gamma.mean(shape, loc=loc, scale=scale)
                    sigma = gamma.std(shape, loc=loc, scale=scale)
                    pdf_values = gamma.pdf(bin_centers, mu, sigma)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))

        elif choice == "Normal":
            params = norm.fit(df["values"])
            dist = norm(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    mean = st.slider("Mean", float(min(values)), float(max(values)), float(np.mean(values)))
                    std = st.slider("Standard deviation", 0.1, float(np.std(values)*3), float(np.std(values)))          
                    fitted_dist = stats.norm(loc=mean, scale=std)
            
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
                
                else: 
                    mu, sigma = norm.fit(values)
                    pdf_values = norm.pdf(bin_centers, mu, sigma)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
           
        elif choice == "Uniform":
            params = uniform.fit(df["values"])
            dist = uniform(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    loc = st.slider("Location", float(min(values)), float(max(values)), float(min(values)))
                    scale = st.slider("Scale", 0.1, float(max(values) - min(values)), float(max(values)-min(values)))          
                    fitted_dist = stats.uniform(loc=loc, scale=scale)
            
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    mu, sigma = uniform.fit(values)
                    pdf_values = uniform.pdf(bin_centers, mu, sigma)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))

        elif choice == "Left-skewed": 
            params = gumbel_l.fit(df["values"])
            dist = gumbel_l(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    alpha = st.slider("Alpha", 0.1, 10.0, 2.0)
                    beta2 = st.slider("Beta", 0.1, 10.0, 5.0)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    scale = st.slider("Scale", 0.1, float(max(values)-min(values)), 1.0)
                    fitted_dist = stats.beta(a=alpha, b=beta2, loc=loc, scale=scale)
            
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
                    
                else:
                    mu, sigma = gumbel_l.fit(values)
                    pdf_values = gumbel_l.pdf(bin_centers, mu, sigma)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
        
        elif choice == "Right-skewed":
            params = gumbel_r.fit(df["values"])
            dist = gumbel_r(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    alpha = st.slider(("Alpha (shape)"), 0.1, 10.0, 2.0)
                    beta = st.slider("Beta (scale)", 0.1, float(max(values)-min(values)), 1.0)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    fitted_dist = stats.gamma(a=alpha, loc=loc, scale=beta)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    mu, sigma = gumbel_r.fit(values)
                    pdf_values = gumbel_r.pdf(bin_centers, mu, sigma)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
        
        elif choice == "Exponential":
            params = expon.fit(df["values"])
            dist = expon(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    scale = st.slider("Scale", 0.1, float(max(values)), float(np.mean(values)))
                    fitted_dist = stats.expon(loc=loc, scale=scale)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    mu, sigma = expon.fit(values)
                    pdf_values = expon.pdf(bin_centers, mu, sigma)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
        
        elif choice == "Beta":
            params = beta.fit(df["values"])
            dist = beta(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)   
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    alpha = st.slider("Alpha", 0.1, 10.0, 2.0)
                    beta = st.slider("Beta", 0.1, 10.0, 5.0)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    scale = st.slider("Scale", 0.1, float(max(values)-min(values)), 1.0)
                    fitted_dist = stats.beta(a=alpha, b=beta, loc=loc, scale=scale)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    a, b, loc, scale = beta.fit(values)
                    pdf_values = beta.pdf(bin_centers, a, b, loc=loc, scale=scale)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
        
        elif choice == "Pareto":
            params = pareto.fit(df["values"])
            dist = pareto(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    b = st.slider("Shape", 0.1, 10.0, 2.0)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    scale = st.slider("Scale", 0.1, float(max(values)-min(values)), 1.0)
                    fitted_dist = stats.pareto(b, loc=loc, scale=scale)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    b, loc, scale = pareto.fit(values)
                    pdf_values = pareto.pdf(bin_centers, b, loc=loc, scale=scale)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
        elif choice == "Lognormal":
            params = lognorm.fit(df["values"])
            dist = lognorm(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    s = st.slider("Shape", 0.1, 2.0, 0.5)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    scale = st.slider("Scale", 0.1, float(max(values)), 1.0)
                    fitted_dist = stats.lognorm(s, loc=loc, scale=scale)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    shape, loc, scale = lognorm.fit(values)
                    pdf_values = lognorm.pdf(bin_centers, shape, loc=loc, scale=scale)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
        
        elif choice == "Weibull_min":
            params = weibull_min.fit(df["values"])
            dist = weibull_min(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    c = st.slider("Shape", 0.1, 2.0, 0.5)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 1.0)
                    scale = st.slider("Scale", 0.1, float(max(values)-min(values)), 1.0)
                    fitted_dist = stats.weibull_min(c, loc=loc, scale=scale)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    c, loc, scale = weibull_min.fit(values)
                    pdf_values = weibull_min.pdf(bin_centers, c, loc=loc, scale=scale)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
      
        elif choice == "Cauchy":
            params = cauchy.fit(df["values"])
            dist = cauchy(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    loc = st.slider("Location", float(min(values)), float(max(values)), float(np.median(values)))
                    scale = st.slider("Scale", 0.1, float(max(values)-min(values)) / 2, 1.0)
                    fitted_dist = stats.cauchy(loc=loc, scale=scale)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    mu, sigma = cauchy.fit(values)
                    pdf_values = cauchy.pdf(bin_centers, mu, sigma)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
     
        elif choice == "Chi2":
            params = chi2.fit(df["values"])
            dist = chi2(*params)
            x = np.linspace(0, df["values"].max(), 100)
            fit = dist.pdf(x)
            axis.plot(x, fit)
            
            with col1:
                manual = st.checkbox("Would you like to use manual fit?")
                if manual:
                    df_param = st.slider("Degrees of Freedom", 1, 20, 2)
                    loc = st.slider("Location", float(min(values)), float(max(values)), 0.0)
                    scale = st.slider("Scale", 0.1, float(max(values)-min(values)), 1.0)
                    fitted_dist = stats.chi2(df=df_param, loc=loc, scale=scale)
                   
                    x_manual = np.linspace(min(values), max(values), 1000)
                    pdf_manual = fitted_dist.pdf(x_manual)
                    
                    axis.clear()
                    axis.hist(df["values"], bins=30, density=True, alpha=0.5, label="Data")
                    axis.plot(x_manual, pdf_manual, 'r-', lw=2, label="Manual Fit")
                    axis.legend()
                    
                    pdf_values = fitted_dist.pdf(bin_centers)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
            
                else:
                    a, loc, scale = chi2.fit(values)
                    pdf_values = chi2.pdf(bin_centers, a, loc=loc, scale=scale)
                    mae = np.mean(np.abs(counts - pdf_values)) # mean absolute error
                    mse = np.mean((counts - pdf_values) ** 2) # mean squared error
                    max_error = np.max(np.abs(counts - pdf_values))
    
        else: 
            st.error("Please enter one of the options as displayed.")
            mae = 0 # mean absolute error
            mse = 0 # mean squared error
            max_error = 0

        st.pyplot(figure)
        
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Maximum Error: {max_error:.4f}")
        

                
        

       
        
        
        
        




