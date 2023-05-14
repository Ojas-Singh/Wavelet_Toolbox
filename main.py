import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
import pywt
import pywt.data
from pywt import wavedec
import plotly.express as px
from bokeh.plotting import figure


def s_table_gen(x,s):
     sindex=[0]
     sxindex=[0]
     dualtime=[]
     value=[s[0]]
     h_diff=[0]
     h_diffmag=[]
     q=1
     for i in range(1,len(s)):
          if s[i-1]!= s[i]:
               sindex.append(q)
               sxindex.append(i)
               dualtime.append(x[sxindex[q]]-x[sxindex[q-1]])
               value.append(s[i])
               h_diff.append(s[i]-s[i-1])
               q+=1

     for i in h_diff:
          h_diffmag.append(np.abs(i))
     dualtime.append(dualtime[-1])
     return [sindex,sxindex,dualtime,value,h_diff,h_diffmag]

def unique(list1):
     unique_list = []
     for x in list1:
          if x not in unique_list:
                    unique_list.append(x)
     return unique_list

def new_table(h,x,s):
     table=s_table_gen(x,s)
     sindex,sxindex,dualtime,value,h_diff,h_diffmag=table
     so= list(zip(h_diffmag,sindex))
     so.sort()
     group=[]
     for i in so:
          if i[0]<h:
               group.append(i[1])
     group.sort()
     o=[]
     for i in group:
          if i !=0:
               o.append([i-1,i])
     xn,sn= s_merge(o,x,s,table)
     ntable= s_table_gen(xn,sn)
     return xn,sn,ntable

def s_merge(k,x,s,table):
     xn=x
     pp=[]
     for i in k:
          a=0
          b=0
          for j in i:
               a+=float(table[2][j])*float(table[3][j])
               b+=table[2][j]
          v2=float(a/b)
          for j in i:
               table[3][j]=v2 
     sn=[]
     for i in table[0]:
          if i!=table[0][-1]:
               for j in range(table[1][i],table[1][i+1]):
                    sn.append(table[3][i])     
          else:
               for j in range(table[1][i],len(x)):
                    sn.append(table[3][i])  
     return xn,sn


#coif17
#'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 


def filter(y,option,t):
     coeffs= pywt.fswavedecn(y,wavelet = option)
     levels= len(coeffs.coeff_slices[0])
     # for i in range(levels):
     #      st.write("Level :",i,"Start :",coeffs.coeff_slices[0][i].start,"Stop",coeffs.coeff_slices[0][i].stop)
     p=[[coeffs.coeff_slices[0][i].start,coeffs.coeff_slices[0][i].stop] for i in range(levels)]
     
     for i in range(levels):
          if not t[i][0] :
               coeffs.coeffs[int(p[i][0]):int(p[i][1])]=0
          else:
               coeffs.coeffs[int(p[i][0]):int(p[i][1])]=coeffs.coeffs[int(p[i][0]):int(p[i][1])]*t[i][1]
     return pywt.fswaverecn(coeffs)


def wave(y,option):
     coeffs= pywt.fswavedecn(y,wavelet = option)
     levels= len(coeffs.coeff_slices[0])
     # for i in range(levels):
     #      st.write("Level :",i,"Start :",coeffs.coeff_slices[0][i].start,"Stop",coeffs.coeff_slices[0][i].stop)
     ilevel= st.slider('Level to Include',1,levels,3)
     p=[[coeffs.coeff_slices[0][i].start,coeffs.coeff_slices[0][i].stop] for i in range(levels)]
     # for i in range(ilevel):
     #      start= float(coeffs.coeff_slices[0][i].start)
     #      stop = float(coeffs.coeff_slices[0][i].stop)
     #      p[i] = st.slider('Range for Level :'+str(i),start,stop,(start,stop),key=i,step =1.0 )
     for i in range(levels):
          if i > ilevel:
               coeffs.coeffs[int(p[i][0]):int(p[i][1])]=0
          else:
               coeffs.coeffs[int(p[i-1][1]):int(p[i][0])]=0
               coeffs.coeffs[int(p[i][1]):int(p[i+1][0])]=0
     return pywt.fswaverecn(coeffs)


def con(levels):
     t=[]
     for i in range(levels):
          a = st.checkbox('Include Level : '+str(i),key=i)
          # if a:
          b=st.slider('Multiplier : ', 0.0, 1.0, 1.0,key=i)
          t.append((a,b))
          # else:
          #      t.append((a,1.0))
     return t

@st.cache
def dataload(uploaded_file):
     data = np.loadtxt(uploaded_file)
     x =data[:, 0]
     y = data[:, 1]
     total_time= x[len(x)-1]-x[0] 
     fs= len(y)
     fz= fs/total_time
     st.write("Total Time: ",total_time,"Sampling Rate :",fz)
     return x,y

def plot(uploaded_file,extra,step):
     x,y=dataload(uploaded_file)
     if extra:
          option = st.selectbox(
     'Which wavelet to use for filtering?',
     ('db38','coif17','sym20','bior6.8','rbio6.8','dmey','haar'))
          # coeffs= pywt.fswavedecn(y,wavelet = option)
          # levels= len(coeffs.coeff_slices[0])
          levels= int(pywt.dwt_max_level(len(x), option)+1)

          t=con(levels)
          y1= filter(y,option,t)
          if st.button('Plot filter',key=1):
               
               psigplot = figure(
               title='Data',
               x_axis_label='Time',
               y_axis_label='Extention')
               psigplot.line(x, y, legend_label='Raw', line_width=0.2,color='black')
               psigplot.line(x, y1[:len(x)], legend_label='Step-Fit', line_width=2,color='red')
               st.bokeh_chart(psigplot, use_container_width=True)    
               DF = pd.DataFrame(zip(x,y1[:len(x)]))
               st.download_button(
               label="Download data",
               data=DF.to_csv(header=None, index=None, sep=' ', mode='a'),
               file_name='filtered.txt',
               mime='text/csv',)
     if step:
          y2= wave(y,'haar')
          ho = st.text_input('H thresold', '2',key=200)
          # h=st.slider('H thresold :',0.0,50.0,0.0,key=0 )
          h=float(ho)
          if st.button('Plot Step-Fit',key=2):
               x0=x
               lp=True
               y0=y2[:len(x)]
               table=[]
               while lp:
                    xe,ye,table=new_table(h,x0,y0)
                    rt=[ye==y0]
                    rt=np.asarray(rt)
                    if rt.all():
                         lp= False
                    x0=xe
                    y0=ye

               # xn,sn,table=new_table(h,x, y2[:len(x)])
               # Bokeh
               p = figure(
               title='Data',
               x_axis_label='Time',
               y_axis_label='Extention')
               p.line(x, y, legend_label='Raw', line_width=0.2)
               p.line(x, y0, legend_label='Step-Fit', line_width=2,color='red')
               st.bokeh_chart(p, use_container_width=True)
               DF = pd.DataFrame(zip(x0,y0))
               st.download_button(
               label="Download data",
               data=DF.to_csv(header=None, index=None, sep=' ', mode='a'),
               file_name='step_fit.txt',
               mime='text/csv',key=300)
               tablef= zip(table[0],table[1],table[2],table[3],table[4],table[5])
               DFt = pd.DataFrame(tablef,columns=['Step_id','strat_time_index','dwell_time','y_value','h diff','h_diff_mag'])
               st.download_button(
               label="Download step dwell data",
               data=DFt.to_csv(header=True,columns=['Step_id','strat_time_index','dwell_time','y_value','h diff','h_diff_mag'] ,index=None, sep=' ', mode='a'),
               file_name='dwell.txt',
               mime='text/csv',key=400)
               



# st.set_page_config(layout="wide")
st.title('Wavelet Toolbox')

uploaded_file = st.file_uploader("Choose a file",key=500)

agree = st.checkbox('Load Dummy Data',key=1200)

if uploaded_file is not None:
     extra = st.checkbox('Filter?',key=600)
     step=st.checkbox('Step-Fit?',key=700)
     plot(uploaded_file,extra,step)
else:
     if agree:
          extra = st.checkbox('Filter?',key=800)
          step=st.checkbox('Step-Fit?',key=900)
          plot('raw.txt',extra,step)
     else:
          st.write("Please Upload a data file to continue!")
# data = np.loadtxt(uploaded_file,dtype=np.float64)
# x =data[:, 0]
# y = data[:, 1]
# st.write(len(x))




# Plotly

# df = pd.DataFrame(list(zip(y, RR[:len(x)])),index =x,columns =['Raw', 'Smooth'])
# fig = px.line(df,template="plotly_dark")
# st.plotly_chart(fig, use_container_width=True)

# Matplotlib

# fig, ax = plt.subplots()
# ax.set_title("Raw Data")
# ax.set_xlabel('Time', labelpad=10)
# ax.set_ylabel('TimeExtention', labelpad=10)
# ax.plot(x, y,linewidth=0.2,color="black")
# ax.plot(x, RR[:len(x)],linewidth=1,color="red")
# st.pyplot(fig)
# final_data = np.asarray(x,dtype=str)
# st.download_button('Download', final_data)
