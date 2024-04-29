import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import altair as alt

import matplotlib.pyplot as plt
import numpy_financial as npf
from mplcursors import cursor
#import io

import plotly.graph_objs as go



st.set_page_config(
    page_title="Solar PV",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded") 

alt.themes.enable("dark")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file, header=None, usecols=range(0, 10), skiprows=range(0, 1))
    return None
    
with st.sidebar:
    st.title('🏂 Investment Analysis')
    
    with st.expander("Upload data"):
        # epandable tree for file upload
        uploaded_file = st.file_uploader("Choose the data file", type=["xlsx", "xls"])

    # Load the data only once and cache the result
    data = load_data(uploaded_file)
            
#%% providing choice of net metering, CFD and RID analysis
    
## function for toggling button 
    def tog_but_ana_type(button_name):
        if button_name == 'net_meter_button' and st.session_state.net_meter_button == True:
            st.session_state.cfd_button = False
            st.session_state.rid_button = False
        if button_name == 'cfd_button' and st.session_state.cfd_button == True:
            st.session_state.net_meter_button = False
            st.session_state.rid_button = False
        if button_name == 'rid_button' and st.session_state.rid_button == True:
            st.session_state.net_meter_button = False
            st.session_state.cfd_button = False

    if 'net_meter_button' not in st.session_state:
        st.session_state.net_meter_button = True

    with st.expander("Analysis type"):
        # a tree for toggle buttons of three analysis types

        net_meter_button = st.toggle('Net metering',key="net_meter_button",on_change=tog_but_ana_type,args=("net_meter_button",))
        cfd_button = st.toggle('2-ways CFD',key="cfd_button",on_change=tog_but_ana_type,args=("cfd_button",))
        RID_button = st.toggle('Simplified RID',key="rid_button",on_change=tog_but_ana_type,args=("rid_button",))
        



#%%




with st.sidebar:
    with st.expander("Parameters"):
        col1, col2, col3 = st.columns([1.6,2.5,3])
        
        with col1:
            # Use st.markdown to display text with custom CSS class for vertical alignment
            st.markdown('<div style="font-size:21px;"><br></div>', unsafe_allow_html=True)
            st.markdown('CUSF')
            st.markdown('<div style="font-size:10px;"><br></div>', unsafe_allow_html=True)
            st.markdown('RID')

        with col2:
            cusf_val = st.number_input("Value", key="cusf_val", value=0.06637, step=0.1)
            RID_val = st.number_input("", key="RID_val", value=0.044, step=0.1,label_visibility="collapsed")
        
        with col3:
            cusf_inc = st.number_input("% increment", key="cusf_inc", value=2.0, step=0.1)
            RID_inc = st.number_input("", key="RID_inc", value=2.0, step=0.1,label_visibility="collapsed")
            
        
        saving_val = st.number_input("Saving price", key="saving_val", value=0.09, step=0.01)
        strike_val = st.number_input("Strike price", key="strike_val", value=0.09, step=0.01)
        plant_size = st.number_input("Plant size", key="plant_size", value=450.0, step=0.1)
        plant_degen = st.number_input("Plant degeneration (%)", key="plant_degen", value=0.55, step=0.001)
        k_e_var = st.number_input("Equity cost (%)", key="k_e_var", value=7.0, step=0.1)
        k_d_var = st.number_input("Interest rate (%)", key="k_d_var", value=3.5, step=0.1)

for i in range(1,2):
    if st.session_state.cfd_button == True:
        method_sel = "2-ways CFD"
    elif st.session_state.rid_button == True:
        method_sel = "RID"
    elif st.session_state.net_meter_button == True:
        method_sel = "Net metering"
    else:
        method_sel = "Net metering"        

        
input_vars_list = np.array([cusf_val,RID_val,cusf_inc,RID_inc,saving_val,
                            strike_val,plant_size,k_e_var,k_d_var,plant_degen])


# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#     background-color: #0099ff;
#     color:#ffffff;
# }
# div.stButton > button:hover {
#     background-color: #00ff00;
#     color:#ff0000;
#     }
# </style>""", unsafe_allow_html=True)
      

#%% function calculate
#@st.cache_data
def calculate_func(input_vars_list,method_sel,data):
    cusf_incr_factor = input_vars_list[2]/100
    RID_incr_factor = input_vars_list[3]/100
    incr_factor = 0.02
    decr_factor = input_vars_list[9]/100
    plant_size = input_vars_list[6] ## KW
    plant_sel = plant_size
    current_yr = 2019
    
    ## selcting strategy for analysis Net metering, 2-ways CFD, RID

    method_nm_com_str = "Net metering"
    method_cfd_com_str = "2-ways CFD"
    method_RID_com_str = "RID"

    ## defining prices and other parameters

    CAPEX_tot = (1543.2-102.8*np.log(plant_sel))*plant_sel

    strike_price = input_vars_list[5]
    RID_price_val = input_vars_list[1]
    saving_price = input_vars_list[4]
    cusf_price_val = input_vars_list[0]
    useful_life = 20
    fiscal_life = 11
    loan_period = 15
    lev_perc = 0.7
    k_e = input_vars_list[7]/100
    k_d = input_vars_list[8]/100
    tax_rate = 0.28
    i_disc = k_e ## becuase of levered

    opex_inv_kw = 60
    opex_inv = opex_inv_kw*plant_sel
    opex_inv_year = 15

    opex_other_kw_yr = 12
    opex_other = opex_other_kw_yr*plant_sel

    insurance_cost_per = 0.003
    insurance_cost = insurance_cost_per*CAPEX_tot
    
    #%%

    ## defining pun and puz for 20 years year 0 is discarded

    pun_t = data.iloc[:,0].values
    puz_t = data.iloc[:,1].values

    no_years = 20

    pun = np.zeros((pun_t.size,no_years+1))
    pun[:,0] = pun_t
    puz = np.zeros((puz_t.size,no_years+1))
    puz[:,0] = puz_t

    for i in range(1,pun.shape[1]):
        pun[:,i] = pun[:,i-1]*(1+incr_factor)
        
    for i in range(1,puz.shape[1]):
        puz[:,i]=puz[:,i-1]*(1+incr_factor)

    pun = pun[:,1:]
    puz = puz[:,1:]
    pun = pun/1000
    puz = puz/1000

    ## defining cusf and RID and saving price

    sav_price = np.zeros((pun.shape[0],pun.shape[1]+1))
    sav_price = sav_price + saving_price
    cusf_price = np.zeros((pun.shape[0],pun.shape[1]+1))
    cusf_price = cusf_price + cusf_price_val
    cfd2_price = np.zeros((pun.shape[0],pun.shape[1]+1))
    cfd2_price = cfd2_price + strike_price
    RID_price = np.zeros((pun.shape[0],pun.shape[1]+1))
    RID_price = RID_price + RID_price_val

    

    for i in range(1,sav_price.shape[1]):
        sav_price[:,i] = sav_price[:,i-1]*(1+incr_factor)

    for i in range(1,cusf_price.shape[1]):
        cusf_price[:,i] = cusf_price[:,i-1]*(1+cusf_incr_factor)

    for i in range(1,RID_price.shape[1]):
        RID_price[:,i] = RID_price[:,i-1]*(1+RID_incr_factor)
        
    sav_price = sav_price[:,1:]
    cusf_price = cusf_price[:,1:]
    RID_price = RID_price[:,1:]
    cfd2_price = cfd2_price[:,1:]

    ### finding where PUZ is greater than RID and replacing 

    ind = puz>RID_price
    RID_price[ind] = puz[ind]


    


    ## consumption and generation and grid feeding and from grid taking

    hourly_cons_y = data.iloc[:,7].values
    hourly_gen_y = data.iloc[:,8].values  ## Wh/kW
    
    hourly_gen_y = hourly_gen_y/1000*plant_sel

    hourly_cons = np.zeros((hourly_cons_y.size,no_years))
    hourly_cons[:,0] = hourly_cons_y
    hourly_gen = np.zeros((hourly_gen_y.size,no_years))
    hourly_gen[:,0] = hourly_gen_y

    for i in range(1,hourly_cons.shape[1]):
        hourly_cons[:,i] = hourly_cons[:,i-1]
        
    for i in range(1,hourly_gen.shape[1]):
        hourly_gen[:,i] = hourly_gen[:,i-1]*(1-decr_factor)
        
    # self consumption

    ind = hourly_gen<=hourly_cons
    self_cons = np.zeros((hourly_gen.shape[0],hourly_gen.shape[1]))
    self_cons[ind] = hourly_gen[ind]
    self_cons[~ind] = hourly_cons[~ind]

    E_i = np.zeros(hourly_gen.shape)
    E_i[~ind] = hourly_gen[~ind] - hourly_cons[~ind]

    E_t = np.zeros(hourly_gen.shape)
    E_t[ind] = hourly_cons[ind] - hourly_gen[ind]

    ## calculating Cei and Ot

    Cei = E_i*puz
    O_t = E_t*pun

    Cei_over_yr = np.sum(Cei,axis=0)
    Cei_over_yr = np.transpose(Cei_over_yr).reshape((1,-1))
    O_t_over_yr = np.sum(O_t,axis=0)
    O_t_over_yr = np.transpose(O_t_over_yr).reshape((1,-1))

    E_t_over_yr = np.sum(E_t,axis=0)
    E_t_over_yr = np.transpose(E_t_over_yr).reshape((1,-1))
    E_i_over_yr = np.sum(E_i,axis=0)
    E_i_over_yr = np.transpose(E_i_over_yr).reshape((1,-1))
    
    #%% amortization schedule and plot

    loan_yr_arr = np.arange(2020,2020+loan_period,1)
    loan_yr_arr = loan_yr_arr.reshape((1,-1))
    loan_prin_arr = np.ones(loan_yr_arr.shape)
    loan_prin_arr = loan_prin_arr*(lev_perc*CAPEX_tot/loan_period)
    loan_prin_arr_cum = (np.cumsum(loan_prin_arr)).reshape((1,-1))
    loan_rem_arr = np.zeros(loan_yr_arr.shape)+lev_perc*CAPEX_tot
    loan_rem_arr = loan_rem_arr-loan_prin_arr_cum
    loan_rem_arr = np.append(lev_perc*CAPEX_tot,loan_rem_arr)
    loan_int_arr = (loan_rem_arr[:-1]*k_d).reshape((1,-1))
    loan_rem_arr = (loan_rem_arr[1:])
    loan_rem_arr = loan_rem_arr.reshape((1,-1))

    fig,ax1 = plt.subplots(1,1,figsize=(7,3))
    ax2 = ax1.twinx()

    bar1= ax1.bar(loan_yr_arr[0,:],loan_prin_arr[0,:],0.4,color='b',label='Principal')
    bar2 = ax1.bar(loan_yr_arr[0,:],loan_int_arr[0,:],0.4,color='r',bottom=loan_prin_arr[0,:],\
            label='Interest')
    line1 = ax2.plot(loan_yr_arr[0,:],loan_rem_arr[0,:],color='k',label='Remaining Loan')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,\
               loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=3)

    ax1.set_ylabel('$')
    ax2.set_ylabel('$')
    plt.title('Amortization Schedule')
    cursor(hover=True)
    plt.xlim(2019,2019+loan_period+1)
    plt.ylim(0,None)
    plt.tight_layout(rect=[0.5, 0, 1, 1])
    
    
    # Create traces
    trace1 = go.Bar(x=loan_yr_arr[0, :], y=loan_prin_arr[0, :], name='Principal')
    trace2 = go.Bar(x=loan_yr_arr[0, :], y=loan_int_arr[0, :], name='Interest')
    title_str = 'Amortization schedule'
    
    layout = go.Layout(barmode='stack', 
                       #title=title_str,
                       width=800,height=300,
                       yaxis=dict(title='$'),
                       margin=dict(
                           l=50,
                           r=50,
                           b=10,
                           t=10,
                           pad=4),
                       #paper_bgcolor="LightSteelBlue"
    )
    
    # Create figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    

    
    

    #%% net metering fee

    energy_quota = np.minimum(Cei_over_yr,O_t_over_yr)
    service_quota = (np.minimum(E_i_over_yr,E_t_over_yr))*cusf_price[0,:]
    surplus_quota = np.maximum(Cei_over_yr-O_t_over_yr,0)
    net_metering_fee = energy_quota + service_quota + surplus_quota

    #%%  revenues

    rev_saving = self_cons*sav_price
    rev_saving_over_yr = np.sum(rev_saving,axis=0)
    rev_saving_over_yr = rev_saving_over_yr.reshape((1,-1))

    rev_cfd = E_i*cfd2_price
    rev_cfd_over_yr = np.sum(rev_cfd,axis=0)
    rev_cfd_over_yr = rev_cfd_over_yr.reshape((1,-1))

    rev_RID = E_i*RID_price
    rev_RID_over_yr = np.sum(rev_RID,axis=0)
    rev_RID_over_yr = rev_RID_over_yr.reshape((1,-1))


    #%% costs

    opex_over_yr = opex_other*np.ones(energy_quota.shape)
    opex_over_yr[0,opex_inv_year-1] = opex_over_yr[0,opex_inv_year-1] + opex_inv
    opex_over_yr = opex_over_yr + insurance_cost

    max_length = max((loan_int_arr.shape)[1],(energy_quota.shape)[1])
    int_over_yr = np.pad(loan_int_arr[0,:], (0, max_length - (loan_int_arr.shape)[1]),\
                                             mode='constant')
    int_over_yr = int_over_yr.reshape((1,-1))
    prin_over_yr = np.pad(loan_prin_arr[0,:], (0, max_length - (loan_prin_arr.shape)[1]),\
                                             mode='constant')
    prin_over_yr = prin_over_yr.reshape((1,-1))

    amort_over_yr = np.zeros((1,fiscal_life))+CAPEX_tot/fiscal_life
    max_length = max((amort_over_yr.shape)[1],(energy_quota.shape)[1])
    amort_over_yr = np.pad(amort_over_yr[0,:], (0, max_length - (amort_over_yr.shape)[1]),\
                                             mode='constant')
    amort_over_yr = amort_over_yr.reshape((1,-1))

    ## calulating EBT for taxes calculation

    rev_tot_netmeter = rev_saving_over_yr + net_metering_fee
    rev_tot_cfd = rev_saving_over_yr + rev_cfd_over_yr
    rev_tot_RID = rev_saving_over_yr + rev_RID_over_yr

    # selection of the method and according calculations

    if method_sel == method_nm_com_str:
        
        EBITDA_over_yr = rev_tot_netmeter - opex_over_yr
        pie_plots_data = np.concatenate((rev_saving_over_yr,energy_quota,service_quota),axis=0)

    elif method_sel == method_cfd_com_str:
        
        EBITDA_over_yr = rev_tot_cfd - opex_over_yr
        pie_plots_data = np.concatenate((rev_saving_over_yr,rev_cfd_over_yr),axis=0)
        
    elif method_sel == method_RID_com_str:
        
        EBITDA_over_yr = rev_tot_RID - opex_over_yr
        pie_plots_data = np.concatenate((rev_saving_over_yr,rev_RID_over_yr),axis=0)
        
    else:
        
        EBITDA_over_yr = rev_tot_netmeter - opex_over_yr
        pie_plots_data = np.concatenate((rev_saving_over_yr,net_metering_fee),axis=0)
        
    EBIT_over_yr = EBITDA_over_yr - amort_over_yr
    EBT_over_yr = EBIT_over_yr - int_over_yr
    tax_over_yr = EBT_over_yr * tax_rate
    tax_over_yr[tax_over_yr<0] = 0
    CF_over_yr = EBITDA_over_yr - prin_over_yr - int_over_yr - tax_over_yr

    yr_arr = np.arange(1,useful_life+1,1)
    yr_arr = yr_arr.reshape((1,-1))
    CF_dis_over_yr = CF_over_yr/(np.power((1+i_disc),yr_arr))
    CF_cum_dis_over_yr = (np.cumsum(CF_dis_over_yr)).reshape(1,-1)

    NPV_over_yr = CF_cum_dis_over_yr - ((1-lev_perc)*CAPEX_tot)
    temp_arr = (np.zeros((1,1))) - ((1-lev_perc)*CAPEX_tot)
    NPV_over_yr = np.concatenate((temp_arr,NPV_over_yr),axis=1)
    yr_arr_with_zer = np.insert(yr_arr,0,values=0,axis=1)

    ## finding pay back year number consider the start zero

    ind = np.argmax(NPV_over_yr>0)
    PBT_yr_no = yr_arr_with_zer[0,ind]

    ## finding IRR and NPV at the end of userful life

    temp_arr = (np.zeros((1,1))) - ((1-lev_perc)*CAPEX_tot)
    CF_over_yr_irr = np.concatenate((temp_arr,CF_over_yr),axis=1)
    IRR_val = round((npf.irr(CF_over_yr_irr[0,:]))*100,2)
    NPV_val = int(NPV_over_yr[0,-1])


    npv_yr_arr = np.arange(2019,2019+useful_life+1,1).reshape(1,-1)
    # Create the Plotly figure
    fig_npv = go.Figure()
    
    # Add a scatter trace for the line plot
    fig_npv.add_trace(go.Scatter(x=npv_yr_arr[0,:], y=NPV_over_yr[0,:], name='NPV',mode='lines', line=dict(color='cyan')))  # Adjust the color as needed
    fig_npv.add_trace(go.Scatter(x=npv_yr_arr[0,:], y=CF_over_yr_irr[0,:], name='CF',mode='lines', line=dict(color='#69ae34')))  # Adjust the color as needed
    
    # Update layout for dark theme
    fig_npv.update_layout(
        #title='Net present value',
        width=800,height=300,
        yaxis=dict(title='$'),
        margin=dict(
            l=50,
            r=20,
            b=10,
            t=10,
            pad=4),
        #paper_bgcolor="LightSteelBlue"
    )



    output_arr = np.array([PBT_yr_no,IRR_val,NPV_val,useful_life])
    

    return output_arr,fig,fig_npv,pie_plots_data,hourly_gen,method_sel,current_yr

#%%

if 'result_f' not in st.session_state:
    st.session_state.result_f = 0


if 'data' in locals() and data is not None:
    with st.sidebar:
        b = st.button("Calculate")
if 'b' in locals():
    if b:
        if 'fig_bar' not in st.session_state:
            st.session_state.fig_bar = go.Figure()
        if 'fig_bar' not in st.session_state:
            st.session_state.fig_npv = go.Figure()
        if 'pie_plots_data_f' not in st.session_state:
            st.session_state.pie_plots_data_f = go.Figure()
        if 'hourly_gen_f' not in st.session_state:
            st.session_state.en_gen_data_f = 0 
        if 'method_sel_r' not in st.session_state:
            st.session_state.method_sel_r = 0 
        if 'current_yr' not in st.session_state:
            st.session_state.current_yr = 0 
        result_f,fig_bar,fig_npv,pie_plots_data_f,hourly_gen_f,method_sel_r,current_yr = calculate_func(input_vars_list,method_sel,data)
        st.session_state.result_f = result_f
        st.session_state.fig_bar = fig_bar
        st.session_state.fig_npv = fig_npv
        st.session_state.pie_plots_data_f = pie_plots_data_f
        st.session_state.hourly_gen_f = hourly_gen_f
        st.session_state.method_sel_r = method_sel_r
        st.session_state.current_yr = current_yr
        
 
output_arr = st.session_state.result_f
if 'fig_bar' in st.session_state:
    fig_bar_plot = st.session_state.fig_bar
    fig_npv_plot = st.session_state.fig_npv

if type(output_arr) is not int:

    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric(label="Pay back time",value=str(int(output_arr[0]))+" years",delta=None)
    with col2:    
        st.metric(label="IRR in "+str(int(st.session_state.current_yr+output_arr[3])), value=str(round(output_arr[1],2))+" %", delta=None)
    with col3:    
        st.metric(label="NPV in "+str(int(st.session_state.current_yr+output_arr[3])), value="$ "+str(int(output_arr[2])), delta=None)
     
    st.markdown(
        """
        <hr style='border: none; border-top: 2px solid #CCCCCC;'> <!-- Adjust the color and thickness as needed -->
        """,
        unsafe_allow_html=True
    )
        
col1, col2 = st.columns([1.25,1])
with col1:
    if 'fig_bar_plot' in locals():
        if fig_bar_plot is not None:
            st.subheader("Amortization Schedule")
            st.plotly_chart(fig_bar_plot,use_container_width=True)
            
with col2:           
    if 'fig_npv_plot' in locals():
        if fig_npv_plot is not None:
            st.subheader("Time history plot")
            st.plotly_chart(fig_npv_plot,use_container_width=True)
            
if 'pie_plots_data_f' in st.session_state:
    st.markdown(
        """
        <hr style='border: none; border-top: 2px solid #CCCCCC;'> <!-- Adjust the color and thickness as needed -->
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1,1.3])
    with col1:
        
        st.subheader("Revenue portfolio")
        pie_plots_data_f_mat = st.session_state.pie_plots_data_f
        yr_sel_arr_1 = st.slider('Select year', 2020, int(2020+output_arr[3]-1),2022,1,key = "rev_yr_slider")
        year_considered_1 = 0 + yr_sel_arr_1 - 2020 # if first year
    
        if st.session_state.method_sel_r == "2-ways CFD":
            labels_pie = ['Energy Savings','CFD incentives']
            colors_pie = ['royalblue','#69ae34']
        elif st.session_state.method_sel_r  == "RID":
            labels_pie = ['Energy Savings','RID incentives']
            colors_pie = ['royalblue','#69ae34']
        elif st.session_state.method_sel_r  == "Net metering":
            labels_pie = ['Energy Savings','Energy quota','Service quota']
            colors_pie = ['royalblue','#69ae34','cain']
        
        
        fig_pie = go.Figure(data=[go.Pie(labels=labels_pie,values=pie_plots_data_f_mat[:,year_considered_1],hole=0.5,marker=dict(colors=colors_pie),textposition='outside')])    
        fig_pie.update_layout(
            #title='Net present value',
            width=400,height=250,
            margin=dict(
                l=50,
                r=20,
                b=10,
                t=30,
                pad=4),
            #paper_bgcolor="LightSteelBlue"
        )
        st.plotly_chart(fig_pie,use_container_width=True)
        
        
    with col2:
        st.subheader("Daily PV potential")
        hourly_gen_f = st.session_state.hourly_gen_f
        yr_sel_arr_2 = st.slider('Select year', 2020, int(2020+output_arr[3]-1),2022,1,key="en_yr_slider")
        year_considered_2 = 0 + yr_sel_arr_2 - 2020 # if first year
        new_mat = hourly_gen_f.reshape(365,24,hourly_gen_f.shape[1]).sum(axis=1)
        days_of_year = np.arange(1, 365 + 1)
        
        fig_en_gen = go.Figure()
        
        # Add a scatter trace for the line plot
        fig_en_gen.add_trace(go.Scatter(x=days_of_year, y=new_mat[:,year_considered_2],mode='lines', line=dict(color='cyan')))  # Adjust the color as needed
    
        
        # Update layout for dark theme
        fig_en_gen.update_layout(
            #title='Net present value',
            width=800,height=250,
            yaxis=dict(title='kWh'),
            margin=dict(
                l=50,
                r=20,
                b=10,
                t=10,
                pad=4),
            #paper_bgcolor="LightSteelBlue"
        )
        #st.subheader("Daily PV potential")
        st.plotly_chart(fig_en_gen,use_container_width=True)
