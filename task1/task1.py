from pywavefront import Wavefront
import numpy as np
import sys
from pyopengltk import OpenGLFrame
from OpenGL.GL import *
from OpenGL.GLU import *
from tkinter import *
from tkinter import filedialog as fd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors
import pandas as pd
from scipy.integrate import odeint

steps = 10
cur_time = 0
time_interval = 200
min_temp = -10
max_temp = 10
initial_conditions = np.array([0, 0, 0, 0, 0])
sol = np.array([[0, 0, 0, 0, 0]])
t = np.linspace(0, time_interval, time_interval * steps + 1)
filename = ""

obj=Wavefront('model2.obj', collect_faces=True)

pointdata = np.array(obj.vertices)

num_of_parts = 5
num_of_part_faces = [384, 384, 384, 224, 192]
part_faces = []
part_vertices = []

cur_num = 0
for i in range(num_of_parts):
    part_faces.append(np.array(obj.mesh_list[0].faces[cur_num : cur_num + num_of_part_faces[i]]))
    cur_num += num_of_part_faces[i]
    part_vertices.append(pointdata[part_faces[i].flat].reshape((-1, 3)))
part_vertices[2], part_vertices[3] = part_vertices[3], part_vertices[2] #в исходном файле располагаются в обратном порядке

def triangle_area(coords):
    a = coords[0] - coords[1]
    b = coords[0] - coords[2]
    c = np.cross(a, b)
    return np.sqrt(c.dot(c)) / 2

def total_area(vertices):
    total_sum = 0
    for i in range(0, len(vertices) , 3):
        total_sum += triangle_area(vertices[i : i + 3])
    return total_sum

def parts_areas(part_vertices):
    res = []
    for i in range(len(part_vertices)):
        res.append(total_area(part_vertices[i]))
    return res

intersections = np.array([[0, 12.25, 0, 0, 0], [12.25, 0, 12.25, 0, 0], [0, 12.5, 0, 3.06, 0], [0, 0, 3.06, 0, 3.06], [0, 0, 0, 3.06, 0]])

def parts_areas_without_intersection(part_vertices, intersections):
    res = parts_areas(part_vertices)
    for i in range(len(res)):
        res[i] -= np.sum(intersections[i])
    return res

areas = parts_areas_without_intersection(part_vertices, intersections)

def load_coefficients(filename):
    global epsilon, c, Q_TC, k, C0, initial_conditions
    coefficients = pd.read_csv(filename, delimiter = ",")
    parameter_values = coefficients['value'].to_list()
    epsilon = parameter_values[0:5]
    c = parameter_values[5:10]
    A = parameter_values[10]
    Q_TC = []
    def Q1(t):
        return A * (20 + 3 * np.cos(t / 4))
    def Q2(t):
        return parameter_values[11]
    def Q3(t):
        return parameter_values[12]
    def Q4(t):
        return parameter_values[13]
    def Q5(t):
        return parameter_values[14]
    Q_TC.append(Q1)
    Q_TC.append(Q2)
    Q_TC.append(Q3)
    Q_TC.append(Q4)
    Q_TC.append(Q5)
    l = np.zeros((5, 5))
    l[0][1] = parameter_values[15]
    l[1][0] = l[0][1]
    l[1][2] = parameter_values[16]
    l[2][1] = l[1][2]
    l[2][3] = parameter_values[17]
    l[3][2] = l[2][3]
    l[3][4]= parameter_values[18]
    l[4][3] = l[3][4]
    k = l * intersections
    C0 = 5.67
    initial_conditions = np.array(parameter_values[19:24])

def pend(T, t):
    f = []
    for i in range(5):
        f.append((np.sum(-k[i] * (T[i] - T)) - epsilon[i] * areas[i] * C0 * ((T[i] / 100.) ** 4) + Q_TC[i](t)) / c[i])
    return f

def temp2rgb(cur_temp):
    return cm.get_cmap("coolwarm")((cur_temp - min_temp) / (max_temp - min_temp))

def leftKey(event):
    glRotatef(5, 0, 1, 0)
def rightKey(event):
    glRotatef(-5, 0, 1, 0)
def upKey(event):
    glRotatef(5, 1, 0, 0)
def downKey(event):
    glRotatef(-5, 1, 0, 0)

def addTimeKey(event):
    global cur_time
    if cur_time < time_interval:
        cur_time += 1
        lbl_cur_time_value.config(text = cur_time)
def reduceTimeKey(event):
    global cur_time
    if cur_time > 0:
        cur_time -= 1
        lbl_cur_time_value.config(text = cur_time)
        
def load_file_with_coefficients():
    filetypes = (
        ('text files', '*.csv'),
        ('All files', '*.*')
    )
    global filename
    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    if filename != '':
        load_coefficients(filename)
        if entry_total_time.get() != "":
            sol_button["state"] = "normal"

def load_temperatures():
    filetypes = (
        ('text files', '*.csv'),
        ('All files', '*.*')
    )
    file = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    if file != "":
        temperatures = pd.read_csv(file, delimiter = ",", index_col = 0)
        global time_interval, t, sol, min_temp, max_temp
        t = np.array(temperatures.index)
        sol = temperatures.to_numpy()
        min_temp = np.min(sol)
        max_temp = np.max(sol)
        time_interval = int(np.max(t))
        update_graph()
        update_colorbar()
        global cur_time
        cur_time = 0
        lbl_cur_time_value.config(text = cur_time)
        entry_total_time.delete(0, END)
        entry_total_time.insert(0, time_interval)
        
def callback(sv):
    global filename
    if sv.get() != "" and filename != "":
        sol_button["state"] = "normal"
    else:
        sol_button["state"] = "disabled"

def sol_ode():
    global time_interval, t, sol, min_temp, max_temp
    time_interval = int(entry_total_time.get())
    t = np.linspace(0, time_interval, time_interval * steps + 1)
    sol = odeint(pend, initial_conditions, t)
    min_temp = np.min(sol)
    max_temp = np.max(sol)
    update_graph()
    update_colorbar()
    global cur_time
    cur_time = 0
    lbl_cur_time_value.config(text = cur_time)
    root.focus()
    pd.DataFrame(sol, index = t, columns = ['T1', 'T2', 'T3', 'T4', 'T5']).to_csv("result.csv", sep=',')

def update_graph():
    global plot1
    plot1.cla()
    plot1.plot(t, sol[:, 0], color = 'red', label = 'T1')
    plot1.plot(t, sol[:, 1], color = 'green', label = 'T2')
    plot1.plot(t, sol[:, 2], color = 'blue', label = 'T3')
    plot1.plot(t, sol[:, 3], color = 'magenta', label = 'T4')
    plot1.plot(t, sol[:, 4], color = 'orange', label = 'T5')
    plot1.set_xlabel('t')
    plot1.set_ylabel('T')
    plot1.set_title("График температуры от времени")
    plot1.legend()
    global canvas1
    canvas1.draw()

def update_colorbar():
    global fig2
    fig2.clear()
    ax2 = fig2.add_subplot(111)
    norm = colors.Normalize(vmin = min_temp, vmax = max_temp)
    cbar = ax2.figure.colorbar(cm.ScalarMappable(norm = norm, cmap='coolwarm'), ax = ax2, pad = .05, extend = 'both', fraction = 1)
    ax2.axis('off')
    global canvas2
    canvas2.draw()

vertex_source = """
varying vec4 vertex_color;
void main(){
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    vertex_color = gl_Color;
}
"""

fragment_source = """
varying vec4 vertex_color;
void main() {
    gl_FragColor = vertex_color;
}
"""

def create_shader(sh_type, sh_source):
    sh = glCreateShader(sh_type)
    glShaderSource(sh, sh_source)
    glCompileShader(sh)
    return sh

class AppOgl(OpenGLFrame):

    def initgl(self):
        vertex = create_shader(GL_VERTEX_SHADER, vertex_source)
        fragment = create_shader(GL_FRAGMENT_SHADER, fragment_source)

        program = glCreateProgram()
        glAttachShader(program, vertex)
        glAttachShader(program, fragment)
        glLinkProgram(program)
        glUseProgram(program)
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(120, 1, 1, 50)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(5, 0, 12, 0, 0, 0, 0, 1, 0)

    def redraw(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        for i in range(num_of_parts):
            glVertexPointer(3, GL_FLOAT, 0, part_vertices[i])
            global cur_time
            color = temp2rgb(sol[cur_time * 10, i])
            glColorPointer(3, GL_FLOAT, 0, np.array(color[:3] * part_vertices[i].shape[0]))
            glDrawArrays(GL_TRIANGLES, 0, len(part_vertices[i]))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
if __name__ == '__main__':
    root = Tk()
    root.wm_title("Расчет тепла")
    
    frame1 = Frame(root)
    frame1.pack(fill = X)
    
    load_coefficients_button = Button(frame1, text = 'Загрузить параметры системы', command = load_file_with_coefficients)
    load_coefficients_button.pack(side = LEFT, padx = 5, pady = 5)
    
    sol_button = Button(frame1, text = 'Произвести расчет', command = sol_ode)
    sol_button["state"] = "disabled"
    sol_button.pack(side = LEFT, padx = 5, pady = 5)
    
    open_button = Button(frame1, text = 'Загрузить температуры', command = load_temperatures)
    open_button.pack(side = LEFT, padx = 5, pady = 5)
    
    frame2 = Frame(root)
    frame2.pack(fill = X)
    
    lbl_total_time = Label(frame2, text = "Введите общее время теплового расчета:")
    lbl_total_time.pack(side = LEFT, padx = 5, pady = 5)
    
    sv_total_time = StringVar()
    sv_total_time.trace("w", lambda name, index, mode, sv = sv_total_time: callback(sv_total_time))
    entry_total_time = Entry(frame2, textvariable = sv_total_time)
    entry_total_time.pack(side = LEFT, padx = 5, pady = 5)
    
    frame3 = Frame(root)
    frame3.pack(fill = X)
    
    lbl_cur_time = Label(frame3, text = "Текущее время:")
    lbl_cur_time.pack(side = LEFT, padx = 5, pady = 5)
    
    lbl_cur_time_value = Label(frame3, text = "")
    lbl_cur_time_value.pack(side = LEFT, padx = 5, pady = 5)
    
    frame4 = Frame(root)
    frame4.pack(fill = X)
    
    global fig
    fig = Figure(figsize = (5, 4), dpi=100)
    global plot1
    plot1 = fig.add_subplot(111)
    plot1.set_xlabel('t')
    plot1.set_ylabel('T')
    plot1.set_title("График температуры от времени")
    global canvas1
    canvas1 = FigureCanvasTkAgg(fig, master = frame4)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side = RIGHT, fill = BOTH, expand = 1)
    
    global fig2
    fig2 = plt.Figure(figsize = (1, 1), dpi=100)
    ax2 = fig2.add_subplot(111)
    norm = colors.Normalize(vmin = min_temp, vmax = max_temp)
    cbar = ax2.figure.colorbar(cm.ScalarMappable(norm = norm, cmap = 'coolwarm'), ax = ax2, pad = .05, extend = 'both', fraction = 1)
    ax2.axis('off')
    global canvas2
    canvas2 = FigureCanvasTkAgg(fig2, master = frame4)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side = RIGHT, fill = Y, expand = 1)

    root.bind('<Left>', leftKey)
    root.bind('<Right>', rightKey)
    root.bind('<Up>', upKey)
    root.bind('<Down>', downKey)
    root.bind('a', reduceTimeKey)
    root.bind('d', addTimeKey)
    
    app = AppOgl(frame4, width = 640, height = 480)
    app.pack(side = LEFT)
    app.animate = 1
    app.mainloop()