import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.write("Hello world! 🌍")
st.write({"key": "value"})
st.write([1, 2, 3])
st.write(123)

3 + 4

'hello world' if False else 'goodbye world'

print('run')

clicked = st.button("Click me!")
print('First:', clicked)

clicked2 = st.button("Click me too!")
print('Second"', clicked2)

st.title("Super Simple Title")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is some text")
st.markdown("This is some _markdown_")

code_example = """
import streamlit as st
import pandas as pd
import numpy as np
st.write("Got lots of data? Great! Streamlit can show [dataframes](https://docs.streamlit.io/develop/api-reference/data) with hundred thousands of rows, images, sparklines – and even supports editing! ✍️")
num_rows = st.slider("Number of rows", 1, 10000, 500)
"""
st.code(code_example, language="python")

import os
st.image(os.path.join(os.getcwd(), "static", "yellow_m3.jpg"), width=500)

st.divider()

st.title("Streamlit Elements Demo")
# Dataframe Section
st.subheader("Dataframe")
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 30, 22, 35],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})
st.dataframe(df, use_container_width=True)

# Data Editor Section (editable dataframe)
st.subheader("Data Editor")
edited_df = st.data_editor(df, use_container_width=True)

# Static Table Section
st.subheader("Static Table")
st.table(df)

# Metrics Section
st.subheader("Metrics")
st.metric(label="Total Rows", value=len(df))
st.metric(label="Average Age", value=df['Age'].mean(), delta=1)

# JSON and Dict Section
st.subheader("JSON and Dict")
sample_dict = {
    'Name': 'Alice',
    'Age': 24,
    'Skills': ['Python', 'Data Science', 'Machine Learning']
}
st.json(sample_dict)

# Also show it as a dict
st.write("Sample Dict:", sample_dict)

# Title
st.title("Streamlit Charts Demo")

# Generate sample data
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

# Area Chart Section
st.subheader("Area Chart")
st.area_chart(chart_data)

# Bar Chart Section
st.subheader("Bar Chart")
st.bar_chart(chart_data)

# Line Chart Section
st.subheader("Line Chart")
st.line_chart(chart_data)

# Scatter Plot Section
st.subheader("Scatter Plot")
scatter_data = pd.DataFrame({
    'x': np.random.rand(50),
    'y': np.random.rand(50),
})
st.scatter_chart(scatter_data)

# Map Section (displaying random points on a map)
st.subheader("Map")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)
st.map(map_data)

# Pyplot Section
st.subheader("Pyplot Chart")
fig, ax = plt.subplots()
ax.plot(chart_data['a'], label='a')
ax.plot(chart_data['b'], label='b')
ax.plot(chart_data['c'], label='c')
ax.set_title("Pyplot Line Chart")
ax.legend()
st.pyplot(fig)

# Title
st.title("Streamlit Form Demo")

# Form to hold the interactive elements
with st.form(key='sample_form'):

    # Text elements
    st.subheader("Text Inputs")
    name = st.text_input("Enter your name")
    feedback = st.text_area("Enter your feedback")

    # Date and Time inputs
    st.subheader("Date and Time Inputs")
    date = st.date_input("Select a date")
    time = st.time_input("Select a time")

    # Selectors
    st.subheader("Selectors")
    option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
    options = st.multiselect("Select multiple options", ["Choice 1", "Choice 2", "Choice 3"])
    slider_value = st.slider("Select a value", 0, 100, 50)
    select_slider_value = st.select_slider("Select a range", options=range(1, 11), value=(3, 7))

    # Toggles and checkboxes
    st.subheader("Toggles and Checkboxes")
    checkbox = st.checkbox("Check me")
    radio = st.radio("Choose one", ["Yes", "No"])

    # Submit button for the form
    submit_button = st.form_submit_button(label='Submit')

# Title
st.title("Streamlit Widgets Demo")
# Widgets Section
st.subheader("Widgets")
st.button("Click Me!")
st.checkbox("Check Me!")
st.radio("Choose One", ["Option 1", "Option 2", "Option 3"])
st.selectbox("Select One", ["Option A", "Option B", "Option C"])
st.multiselect("Select Multiple", ["Choice 1", "Choice 2", "Choice 3"])
st.slider("Slide Me!", 0, 100, 50)
st.select_slider("Select a Range", options=range(1, 11), value=(3, 7))
st.text_input("Type Something")
st.text_area("Type More")
st.date_input("Pick a Date")
st.time_input("Pick a Time")
st.file_uploader("Upload a File")
st.color_picker("Pick a Color")

# Title
st.title("Streamlit Layouts Demo")
# Layouts Section
st.subheader("Layouts")
col1, col2 = st.columns(2)
col1.header("Column 1")
col1.write("This is the first column.")
col2.header("Column 2")
col2.write("This is the second column.")
col3, col4 = st.columns(2)
col3.header("Column 3")
col3.write("This is the third column.")
col4.header("Column 4")
col4.write("This is the fourth column.")
# Expander Section
with st.expander("See More"):
    st.write("This is some additional content inside an expander.")
    st.line_chart(np.random.randn(10, 2))
# Sidebar Section
st.sidebar.title("Sidebar")
st.sidebar.write("This is the sidebar.")
st.sidebar.button("Click Me!", key="sidebar_button")
st.sidebar.checkbox("Check Me!", key="sidebar_checkbox")
# Footer Section
st.write("This is the footer.")

# Title
st.title("User Information Form")

with st.form(key='user_info_form'):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)

    print("Name:", name)
    print("Age:", age)
    st.form_submit_button("Submit")

if name == 'Mike':
    st.success("Hello Mike!")
elif name == 'Alice':
    st.success("Hello Alice!")
elif name == 'Bob':
    st.success("Hello Bob!")
else:
    st.success("Hello Stranger!")


st.title("Streamlit Info Form 2")
from datetime import datetime

form_values = {
    "name": "None",
    "height": "None",
    "dob": "None"
}

min_date = datetime(1900, 1, 1)
max_date = datetime.now()

with st.form(key='user_info_form_2'):
    form_values["name"] = st.text_input("Name")
    form_values["height"] = st.number_input("Height (cm)", min_value=0, max_value=300)
    form_values["dob"] = st.date_input("Date of Birth", max_value=max_date, min_value=min_date)

    submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        print("after submit")
        if not all(form_values.values()):
            print("in if")
            st.warning("Please fill in all fields.")
        else:
            st.balloons()
            st.write('### Info')
            st.success("Form submitted successfully!")
            for (key, value) in form_values.items():
                st.write(f"{key}: {value}")


# Title
st.title("Streamlit User Info Form 3")

with st.form(key='user_info_form_3', clear_on_submit=True):
    name1 = st.text_input("First Name")
    birth_date = st.date_input("Birth Date", min_value=datetime(1900, 1, 1), max_value=datetime.today())

    if birth_date:
        age = (datetime.today().year - birth_date.year)
        if birth_date.month > datetime.today().month or (birth_date.month == datetime.today().month and birth_date.day > datetime.today().day):
            age -= 1
        st.write(f"Your age is: {age} years")

    submit_button1 = st.form_submit_button(label="Submit")

    if submit_button1:
        if not name1 or not birth_date:
            st.warning("Please fill in all fields.")
        else:
            st.success(f"Thank you, {name1}! Your age is {age} years.")
            st.balloons()

# Session State Example
st.title("Session State Example")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("Increment Counter"):
    st.session_state.counter += 1
    st.success(f"Counter: {st.session_state.counter}")

if st.button("Reset Counter"):
    st.session_state.counter = 0
else:
    st.write(f"Counter did not reset")

st.write("Session state counter:", st.session_state.counter)


# Callbacks Example
st.title("Callbacks Example")

if "step" not in st.session_state:
    st.session_state.step = 1
    
if "info" not in st.session_state:
    st.session_state.info = {}

if st.session_state.step == 1:
    st.header("Part 1: Info")

    name = st.text_input("Name", value=st.session_state.info.get("name", ""))

    if st.button("Next"):
        st.session_state.info["name"] = name
        st.session_state.step = 2


elif st.session_state.step == 2:
    st.header("Part 2: Review")

    st.subheader("Please review your information:")
    st.write(f"**Name**: {st.session_state.info.get('name', '')}")

    if st.button("Submit"):
        st.success("Form submitted successfully!")
        st.balloons()
        st.session_state.info = {}

    if st.button("Back"):
        st.session_state.step = 1