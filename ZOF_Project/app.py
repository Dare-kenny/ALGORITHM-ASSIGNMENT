import streamlit as st
import numpy as np, pandas as pd, sympy as sp

st.set_page_config(page_title="ZOF Solver", page_icon="🧮")
st.title("🧮 ZOF — Zero of Functions Solver")

x = sp.symbols('x')
def make_func(expr): return sp.lambdify(x, sp.sympify(expr), 'numpy')
def make_f_and_df(expr): e = sp.sympify(expr); return sp.lambdify(x, e, 'numpy'), sp.lambdify(x, sp.diff(e, x), 'numpy')

# Methods
def bisection(f,a,b,tol,n):
    fa,fb=f(a),f(b)
    if fa*fb>0: raise ValueError("f(a),f(b) same sign")
    r,old=[],None
    for i in range(1,n+1):
        c=(a+b)/2; fc=f(c); err=np.nan if old is None else abs(c-old)
        r.append(dict(i=i,a=a,b=b,x=c,fx=fc,err=err))
        if abs(fc)<tol or (old and err<tol): return c,r
        if fa*fc<0:b,fb=c,fc
        else:a,fa=c,fc
        old=c
    return c,r

def regula_falsi(f,a,b,tol,n):
    fa,fb=f(a),f(b)
    if fa*fb>0: raise ValueError("f(a),f(b) same sign")
    r,old=[],None
    for i in range(1,n+1):
        c=(a*fb-b*fa)/(fb-fa); fc=f(c); err=np.nan if old is None else abs(c-old)
        r.append(dict(i=i,a=a,b=b,x=c,fx=fc,err=err))
        if abs(fc)<tol or (old and err<tol): return c,r
        if fa*fc<0:b,fb=c,fc
        else:a,fa=c,fc
        old=c
    return c,r

def secant(f,x0,x1,tol,n):
    r=[]
    for i in range(1,n+1):
        f0,f1=f(x0),f(x1)
        if f1==f0: raise ZeroDivisionError
        x2=x1-f1*(x1-x0)/(f1-f0)
        err=abs(x2-x1)
        r.append(dict(i=i,x=x2,fx=f(x2),err=err))
        if abs(f(x2))<tol or err<tol:return x2,r
        x0,x1=x1,x2
    return x2,r

def newton(f,df,x0,tol,n):
    r=[]; x=x0
    for i in range(1,n+1):
        fx,dfx=f(x),df(x)
        if dfx==0: raise ZeroDivisionError
        xn=x-fx/dfx; err=abs(xn-x)
        r.append(dict(i=i,x=xn,fx=f(xn),err=err))
        if abs(f(xn))<tol or err<tol:return xn,r
        x=xn
    return x,r

def fixed_point(g,x0,tol,n):
    r=[]; x=x0
    for i in range(1,n+1):
        xn=g(x); err=abs(xn-x)
        r.append(dict(i=i,x=xn,err=err))
        if err<tol:return xn,r
        x=xn
    return x,r

def modified_secant(f,x0,delta,tol,n):
    r=[]; x=x0
    for i in range(1,n+1):
        fx=f(x); denom=f(x+delta*x)-fx
        if denom==0: raise ZeroDivisionError
        xn=x-fx*(delta*x)/denom; err=abs(xn-x)
        r.append(dict(i=i,x=xn,fx=f(xn),err=err))
        if abs(f(xn))<tol or err<tol:return xn,r
        x=xn
    return x,r

# UI
method=st.selectbox("Method",["Bisection","Regula Falsi","Secant","Newton","Fixed Point","Modified Secant"])
tol=st.number_input("Tolerance",value=1e-6,format="%.1e")
it=st.number_input("Max Iterations",value=50,step=1)

if method=="Fixed Point":
    g_str=st.text_input("g(x)","(x+2)**(1/3)")
    x0=st.number_input("x0",value=1.0)
else:
    f_str=st.text_input("f(x)","x**3 - x - 2")
    if method in ["Bisection","Regula Falsi"]:
        a,b=st.number_input("a",value=1.0),st.number_input("b",value=2.0)
    elif method=="Secant":
        x0,x1=st.number_input("x0",value=1.0),st.number_input("x1",value=2.0)
    elif method=="Newton":
        x0=st.number_input("x0",value=1.0)
    elif method=="Modified Secant":
        x0=st.number_input("x0",value=1.0)
        delta=st.number_input("delta",value=1e-6,format="%.1e")

if st.button("Compute"):
    try:
        if method=="Fixed Point":
            g=make_func(g_str)
            root,rows=fixed_point(g,x0,tol,int(it))
            df=pd.DataFrame(rows); st.success(f"Root ≈ {root}"); st.dataframe(df)
        else:
            f,dfunc=make_f_and_df(f_str)
            if method=="Bisection": root,rows=bisection(f,a,b,tol,int(it))
            elif method=="Regula Falsi": root,rows=regula_falsi(f,a,b,tol,int(it))
            elif method=="Secant": root,rows=secant(f,x0,x1,tol,int(it))
            elif method=="Newton": root,rows=newton(f,dfunc,x0,tol,int(it))
            elif method=="Modified Secant": root,rows=modified_secant(f,x0,delta,tol,int(it))
            df=pd.DataFrame(rows); st.success(f"Root ≈ {root}"); st.dataframe(df)
        st.download_button("Download CSV",df.to_csv(index=False),file_name="iterations.csv")
    except Exception as e: st.error(str(e))
