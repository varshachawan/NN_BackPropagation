# Chawan, Varsha Rani
# 1001-553-524
# 2018-11-25
# Assignment-05-01

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
import Chawan_05_02
import matplotlib.pyplot as plt
from sklearn import preprocessing


class MainWindow(tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    Chawan Varsha Rani
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # set the properties of the row and columns in the master frame
        self.rowconfigure(0, weight=1, minsize=500)
        self.columnconfigure(0, weight=1, minsize=500)
        self.master_frame.rowconfigure(2, weight=10, minsize=100, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
        # Create an object for plotting graphs in the left frame
        self.left_frame = tk.Frame(self.master_frame)
        self.left_frame.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_error = Back_propagation(self, self.left_frame, debug_print_flag=self.debug_print_flag)


class Back_propagation:
    """
    This class creates and controls the sliders , buttons , drop down in the frame which
    are used to display decision boundary and generate samples and train .
    """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.alpha = 0.1
        self.lamda = 0.01
        self.nodes = 100
        self.samples = 200
        self.classes = 4
        self.activation_function = "Relu"
        self.generated_data = "s_curve"

        self.train_data, self.train_label = Chawan_05_02.generate_data(self.generated_data, self.samples, self.classes)

        self.w_hid, self.b_hid, self.w_out, self.b_out = self.reset_weights(self.nodes, self.classes)

        #########################################################################
        #  TesorFlow variables
        #########################################################################

        tf.reset_default_graph()
        # self.X = tf.placeholder('float', (self.samples, 2))  # 1x2
        # self.Y = tf.placeholder('float', (self.samples, self.classes))
        self.X = tf.placeholder('float', (None, 2))  # 1x2
        self.Y = tf.placeholder('float', (None,None))
        self.W_hidden = tf.Variable(tf.random_normal((2, self.nodes)))  # hiddenx2 2x100
        self.b_hidden = tf.Variable(tf.random_normal((self.nodes,)))  # 100
        self.W_output = tf.Variable(tf.random_normal((self.nodes, self.classes)))  # 100x4
        self.b_output = tf.Variable(tf.random_normal((self.classes,)))

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_title("")
        self.axes.set_title("")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the control widgets such as sliders ,buttons and dropdowns
        #########################################################################
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.000, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF", label="Learning Rate",
                                     command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.lamda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF", label="Weight regularization",
                                     command=lambda event: self.lamda_slider_callback())
        self.lamda_slider.set(self.lamda)
        self.lamda_slider.bind("<ButtonRelease-1>", lambda event: self.lamda_slider_callback())
        self.lamda_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.hidden_nodes_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                     from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                     label="Nodes in Hidden Layer",
                                     command=lambda event: self.hidden_nodes_slider_callback())
        self.hidden_nodes_slider.set(self.nodes)
        self.hidden_nodes_slider.bind("<ButtonRelease-1>", lambda event: self.hidden_nodes_slider_callback())
        self.hidden_nodes_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.samples_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                            from_=4, to_=1000, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF",
                                            label="Number of Samples",
                                            command=lambda event: self.samples_slider_callback())
        self.samples_slider.set(self.samples)
        self.samples_slider.bind("<ButtonRelease-1>", lambda event: self.samples_slider_callback())
        self.samples_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        self.classes_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                      from_=2, to_=10, resolution=1, bg="#DDDDDD",
                                      activebackground="#FF0000", highlightcolor="#00FFFF",
                                      label="Number of Classes",
                                      command=lambda event: self.classes_slider_callback())
        self.classes_slider.set(self.classes)
        self.classes_slider.bind("<ButtonRelease-1>", lambda event: self.classes_slider_callback())
        self.classes_slider.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)


        self.adjust_weight_button = tk.Button(self.controls_frame, text="Train", width=16,
                                              command=self.adjust_weight_button_callback)
        self.adjust_weight_button.grid(row=1, column=0)

        self.randomize_weight_button = tk.Button(self.controls_frame, text="Reset Weights", width=16,
                                                 command=self.randomize_weight_button_callback)
        self.randomize_weight_button.grid(row=1, column=1)

        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################

        self.label_for_activation_function = tk.Label(self.controls_frame, text="Hidden Layer Transfer Function:",
                                                      justify="center")
        self.label_for_activation_function.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                          "Relu", "Sigmoid",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Relu")
        self.activation_function_dropdown.grid(row=1, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        self.generated_data_label = tk.Label(self.controls_frame, text="Type of generated data:",
                                                  justify="center")
        self.generated_data_label.grid(row=1, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
        self.generated_data_variable = tk.StringVar()
        self.generated_data_drop_down = tk.OptionMenu(self.controls_frame, self.generated_data_variable,
                                                       "s_curve", "blobs",
                                                       "swiss_roll","moons", command=lambda
                event: self.generated_data_dropdown_callback())


        self.generated_data_variable.set("s_curve")
        self.generated_data_drop_down.grid(row=1, column=5, sticky=tk.N + tk.E + tk.S + tk.W)



    def plot_decision_boundary(self, sess):
        #########################################################################
        #  Freeze the weights and calculate o/p , find the index and plot
        #########################################################################
        self.axes.cla()
        x_min, x_max = self.train_data[:, 0].min() - .25, self.train_data[:, 0].max() + .25
        y_min, y_max = self.train_data[:, 1].min() - .25, self.train_data[:, 1].max() + .25
        resolution = 100
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(xs, ys)
        mesh_input = (np.c_[xx.ravel(), yy.ravel()]).astype(np.float32)
        h_net = tf.matmul(mesh_input, self.w_hid) + self.b_hid
        if self.activation_function == 'Relu':
            h_output = tf.nn.relu(h_net)
        elif self.activation_function == 'Sigmoid':
            h_output = tf.nn.sigmoid(h_net)
        o_net = tf.matmul(h_output, self.w_out) + self.b_out
        pred_idx = tf.argmax(o_net, axis=1)
        pred_idx = tf.reshape(pred_idx, xx.shape)

        pred_idx_copy = sess.run(pred_idx)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["r", "darkblue", "sienna", "gold", "green",
                                                                        "chartreuse", "orange", "grey", "darkmagenta",
                                                                        "c"])

        self.axes.pcolormesh(xx, yy, pred_idx_copy, cmap=plt.cm.Spectral)
        self.axes.set_xlabel('Input1')
        self.axes.set_ylabel('Input2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        #########################################################################
        # Plot the input sample points
        #########################################################################

        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], c=self.train_label, cmap=cmap)
        self.axes.xaxis.set_visible(True)
        plt.title(self.activation_function)
        plt.title(self.generated_data)

        self.canvas.draw()

    def train_model(self):
        #########################################################################
        # train the model by using cross entropy softmax function
        #########################################################################
        epoch = 10
        hidden_net = tf.matmul(self.X ,self.W_hidden ) + self.b_hidden
        if self.activation_function == 'Relu':
            hidden_output = tf.nn.relu(hidden_net)
        elif self.activation_function == 'Sigmoid':
            hidden_output = tf.nn.sigmoid(hidden_net)
        out_net = tf.matmul(hidden_output,self.W_output) + self.b_output

        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out_net, labels=self.Y)

        ############################################################################
        # Loss function with L2 Regularization with lambda
        ############################################################################
        loss = tf.reduce_mean(entropy)
        regularizers = tf.nn.l2_loss(self.W_hidden) + tf.nn.l2_loss(self.W_output)

        loss = tf.reduce_mean(loss + self.lamda * regularizers)

        train_op = tf.train.GradientDescentOptimizer(self.alpha).minimize(loss)

        for e in range(epoch):
            #####################################################
            #  Binarizing the input labels (one hot notation)
            ######################################################
            self.lb = preprocessing.LabelBinarizer()
            self.lb.fit(self.train_label)
            self.lb.classes_
            label_Y = self.lb.transform(self.train_label)
            label_Y = label_Y.astype(float)

            self.sess.run(train_op, feed_dict={self.X:self.train_data, self.Y:label_Y})

            self.w_hid = self.sess.run(self.W_hidden)
            self.b_hid = self.sess.run( self.b_hidden)
            self.w_out = self.sess.run(self.W_output)
            self.b_out = self.sess.run(self.b_output)

            self.plot_decision_boundary(self.sess)

    def reset_weights(self,nodes,classes):
        #########################################################################
        #  Reset the weights to random values
        #########################################################################
        w_hid_temp = (np.random.uniform(-0.001, 0.001,(2,nodes))).astype(np.float32)
        b_hid_temp = (np.random.uniform(-0.001, 0.001,(nodes,))).astype(np.float32)
        w_out_temp = (np.random.uniform(-0.001, 0.001,(nodes,classes))).astype(np.float32)
        b_out_temp = (np.random.uniform(-0.001, 0.001,(classes,))).astype(np.float32)
        self.w_hid = w_hid_temp
        self.b_hid = b_hid_temp
        self.w_out = w_out_temp
        self.b_out = b_out_temp

        return w_hid_temp, b_hid_temp, w_out_temp, b_out_temp

    def reset_tensor_variables(self):
        #########################################################################
        #  Reset the tensors and reinitialize to random values
        #########################################################################
        tf.reset_default_graph()
        self.X = tf.placeholder('float', (None, 2))  # 1x2
        self.Y = tf.placeholder('float', (None, None))
        # self.X = tf.placeholder('float', (self.samples, 2))  # 1x2
        # self.Y = tf.placeholder('float', (self.samples, self.classes))
        self.W_hidden = tf.Variable(tf.random_normal((2, self.nodes)))  # hiddenx2 2x100
        self.b_hidden = tf.Variable(tf.random_normal((self.nodes,)))  # 100
        self.W_output = tf.Variable(tf.random_normal((self.nodes, self.classes)))  # 100x4
        self.b_output = tf.Variable(tf.random_normal((self.classes,)))
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)


    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())


    def lamda_slider_callback(self):
        self.lamda = np.float(self.lamda_slider.get())


    def hidden_nodes_slider_callback(self):
        self.nodes = self.hidden_nodes_slider.get()
        self.reset_weights(self.nodes,self.classes)
        self.reset_tensor_variables()
        self.plot_decision_boundary(self.sess)


    def samples_slider_callback(self):
        self.samples = self.samples_slider.get()
        self.train_data, self.train_label = Chawan_05_02.generate_data(self.generated_data, self.samples, self.classes)
        self.reset_weights(self.nodes, self.classes)
        self.reset_tensor_variables()
        self.plot_decision_boundary(self.sess)


    def classes_slider_callback(self):
        self.classes = self.classes_slider.get()
        self.train_data, self.train_label = Chawan_05_02.generate_data(self.generated_data, self.samples, self.classes)
        self.reset_weights(self.nodes, self.classes)
        self.reset_tensor_variables()
        self.plot_decision_boundary(self.sess)


    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.reset_weights(self.nodes, self.classes)
        self.plot_decision_boundary(self.sess)


    def generated_data_dropdown_callback(self):
        self.generated_data = self.generated_data_variable.get()
        self.train_data, self.train_label = Chawan_05_02.generate_data(self.generated_data, self.samples, self.classes)
        self.reset_weights(self.nodes, self.classes)
        self.plot_decision_boundary(self.sess)


    def adjust_weight_button_callback(self):
        self.train_model()


    def randomize_weight_button_callback(self):
        self.reset_weights(self.nodes, self.classes)
        self.reset_tensor_variables()
        self.plot_decision_boundary(self.sess)


        #########################################################################
        #  Logic to close Main Window
        #########################################################################

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()


main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Assignment_05 --  Chawan')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()