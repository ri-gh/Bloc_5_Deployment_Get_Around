from turtle import delay
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pip
import streamlit as st

#to be able to read xlsx file
pip.main(["install", "openpyxl"])

st.set_page_config(
    page_title="Delay Analysis Get Around",
    page_icon="💸 ",
    layout="wide"
)

st.title("⏰💸 Delay Analysis for 'Get Around' App 🚗")


st.markdown("""
    Welcome to this dashboard, we are going to analyze  delays and their consequences for next user,
    what could be the solutions to minimize issues for next customer:
    * threshold between 2 rentals: how long should the minimum delay be?
    * scope of our feature: should we enable the feature for all cars? only Connect cars?

    You can find our `rental prices' predictions API`  🤖 here :  https://fastapilast.herokuapp.com/.
""")

@st.cache (allow_output_mutation=True)
def load_data(nrows):
    data = pd.read_excel('get_around_delay_analysis.xlsx',nrows=nrows)
    return data

data = load_data(18000)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.markdown("---")

st.subheader('Check in type Analysis')
check_in_type = data['checkin_type'].unique()
check_in = st.selectbox("Which type of check in would you like to see?", check_in_type)

###number of delay for mobile:
booking_type = len(data[data['checkin_type'] == check_in])/len(data)*100
delay = len(data[(data['checkin_type'] == check_in) & (data['delay_at_checkout_in_minutes'] > 0)])/len(data)*100
cancellation = len(data[(data['checkin_type'] == check_in) & (data['state'] == 'canceled')])/len(data)*100
ended = len(data[(data['checkin_type'] == check_in) & (data['state'] == 'ended')])/len(data)*100


content, empty_space = st.columns([3, 2])
with empty_space:
    st.empty()
with content:
    st.metric("Percentage of '{}' rentals".format(check_in),"{} %".format(np.round(booking_type,2)))
    st.metric("Percentage of delay for '{}' rentals".format(check_in),"{} %".format(np.round(delay,2)))

st.markdown("  ")

#to have the global view of rentals check in type
fig=px.bar(data,x=data['checkin_type'].value_counts(),y=data['checkin_type'].unique(),orientation='h',
color=data['checkin_type'].unique(),width=800, height=400)
fig.update_layout(
        title="Global view of rentals check in type",
    xaxis_title="Count",
    yaxis_title="Rental type",
    font=dict(family="Arial, monospace",
        size=18,
        ))

st.write(fig)

st.markdown("---")

st.subheader("Global view of rentals state:")
st.markdown("* We have {}% of cancellation".format(np.round((len(data[data['state'] == 'canceled'])/len(data['state'])*100)),2))
st.markdown("* We have {}% of ended rental".format(np.round((len(data[data['state'] == 'ended'])/len(data['state'])*100)),2))

st.markdown("  ")

st.markdown("  ")

df1 = data['state'].value_counts()
st.bar_chart(df1)

st.markdown("  ")

check_in_type1 = data['checkin_type'].unique()
check_in1 = st.selectbox("Which type of check in would you like to review?", check_in_type1)
content, empty_space = st.columns([3, 2])

cancelpertype = len(data[(data['checkin_type'] == check_in1) & (data['state'] == 'canceled')]) /len(data[data['checkin_type'] == check_in1])*100
endedpertype = len(data[(data['checkin_type'] == check_in1) & (data['state'] == 'ended')]) /len(data[data['checkin_type'] == check_in1])*100

with empty_space:
    st.empty()
with content:
    st.metric("Percentage of cancelled rentals for '{}' rental type".format(check_in1),"{} %".format(np.round(cancelpertype,2)))
    st.metric("Percentage of ended rental for '{}' rental type".format(check_in1),"{} %".format(np.round(endedpertype,2)))
   
st.markdown("  ")

#pie chart of rentals per check in type 
pie_chart_data = data.groupby('state')['checkin_type'].value_counts()
fig1, ax1 = plt.subplots(figsize=(3,3))
ax1.pie(pie_chart_data.values, labels=pie_chart_data.index,
        textprops={'color':'white'},
        autopct='%1.1f%%',
       shadow=True, 
       startangle=-15,
        explode = (0.2,0.2,0,0),
       radius=1.5)
fig1.set_facecolor('black')
plt.legend(bbox_to_anchor=(1.1, 1.05),loc = 'lower left', fontsize = 6)
plt.title('Global state of rentals by kind of booking', x= 0, y= 1.3, color ='white', fontsize=9)
plt.show()

st.pyplot(fig1) 


st.subheader('Delay Analysis ⏰')

st.subheader("Global view of delays")

@st.cache
def load_data5(nrows):
    data['late/on time'] = data['delay_at_checkout_in_minutes'].apply(lambda x :'late' if x > 0 else 'on time')
    mask = data['delay_at_checkout_in_minutes'].notnull()
    df = data.loc[mask].reset_index(drop = True)
    return df

df = load_data5(18000)

w = df['late/on time'].value_counts()

st.markdown("* The global percentage of delayed check out is {}% ".format(np.round(len(data[data['delay_at_checkout_in_minutes'] > 0])/len(data[data['delay_at_checkout_in_minutes'].notnull()])*100),2))

@st.cache
def load_datal(nrows):
    tf = data[data['delay_at_checkout_in_minutes'] > 0].reset_index(drop = True)
    tf['delay_at_checkout_in_minutes'] = tf['delay_at_checkout_in_minutes'][np.abs(tf['delay_at_checkout_in_minutes']-tf['delay_at_checkout_in_minutes'].mean())<=(3*tf['delay_at_checkout_in_minutes'].std())]
    return tf
tf = load_data5(18000)

#to have the bar chart of the late and on time check out rental
st.bar_chart(w)
a= (np.round((tf['delay_at_checkout_in_minutes'].mean()),2))
pd.to_datetime(64.58 , unit="%H%M").strftime('%H:%M') #mean delay was of 64.58 minutes we converted it to hour/minutes
st.markdown(' * The average time delay is {} (in hour) '.format((pd.to_datetime(a , unit="%H%M").strftime('%H:%M'))))

@st.cache
def load_datax(nrows):
    dfx = data[data['delay_at_checkout_in_minutes'] > 0]
    return dfx
dfx = load_datax(18000)

st.markdown("* Check in type of delayed checked out people :")

#pie chart of Check in type of delayed checked out people
pie_chart_data1 = dfx['checkin_type'].value_counts()
fig2, ax2 = plt.subplots(figsize=(2.5,2.5))
ax2.pie(pie_chart_data1.values, labels=pie_chart_data1.index,
        textprops={'color':'white'},
        autopct='%1.1f%%',
        startangle=40,
        explode = (0.2,0.2),
       radius=1.5)
fig2.set_facecolor('black')
plt.legend(bbox_to_anchor=(1.1, 1.05),loc = 'lower left', fontsize = 6)
plt.show()

st.pyplot(fig2)

#we want a df with only the late checked out rentals:
@st.cache
def load_data1(nrows):
    data_retard = (df[df['late/on time'] == 'late'].reset_index(drop = True))
    return data_retard

data_retard = load_data1(18000)

if st.checkbox('Show raw list of late drivers'):
    st.subheader('Raw delays data')
    st.write(data_retard)

st.markdown("---")

st.subheader('Analysis of delay s consequences')


retardataires_list = []
for i in range(len(data_retard)):
    retardataires_list.append(data_retard['rental_id'][i])


#we want to analyze behaviour of people after getting their car with delay due to late previous check out
@st.cache
def load_data2(nrows):
    dfa = (data[data['previous_ended_rental_id'].isin(retardataires_list)].reset_index(drop= True))
    return dfa

dfa=load_data2(18000)

if st.checkbox('Show raw data of driver after getting the car with delay'):
    st.subheader('Raw data of drivers getting their car with delay')
    st.write(dfa)


st.markdown("* There are {} delayed check out after getting his rental with delay ".format(len(dfa[dfa['late/on time'] == 'late'])))
dfa.dropna()
b = dfa['late/on time'].value_counts()
st.bar_chart(b)




st.markdown("* There are {} cancellations after a delayed check out ".format(len(dfa[dfa['state'] == 'canceled'])))

c = dfa['state'].value_counts()
st.bar_chart(c)


st.subheader("Delay threshold & feature's scope 🤓:")
data = data[(data['delay_at_checkout_in_minutes'] >= 120) & (data['checkin_type'] == 'mobile')]
st.caption("- We choose to take a threshold of ***2 hours*** & to apply it on the ***'mobile'*** car rental scope")
st.caption("- We will now be able to handle ***{}***  problematic cases (late check out /mobile check in)".format(len(data)))
 

### Side bar 
st.sidebar.header("Dashboard summary")
st.sidebar.markdown("""
    * [Check in type Analysis](#check-in-type-analysis)
    * [Global view of rentals state](#global-view-of-rentals-state)
    * [Delay Analysis ⏰](#delay-analysis)
    * [Analysis of delay s consequences](#analysis-of-delay-s-consequences)
    * [Delay threshold & feature's scope 🤓](#delay-threshold-feature-s-scope)
    """)
e = st.sidebar.empty()
e.write("")





