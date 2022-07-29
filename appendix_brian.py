import random
import numpy as np
from brian2 import *
from tqdm import tqdm
from struct import unpack
from sklearn import metrics


def load_mnist_data(mode='train', path='../mnist_data/', num_samples=10000):
	#Get image data
	images = open(path+('train' if mode == 'train' else 't10k')+'-images.idx3-ubyte', mode='rb')
	images.read(4) #skip magic number
	num_img = unpack('>I', images.read(4))[0]
	rows = unpack('>I', images.read(4))[0] #should be 28
	cols = unpack('>I', images.read(4))[0] #should be 28

	#Get labels data
	labels = open(path+('train' if mode == 'train' else 't10k')+'-labels.idx1-ubyte', mode='rb')
	labels.read(4) #skip magic number
	num_lbl = unpack('>I', labels.read(4))[0]

	#Setup data arrays
	x = np.zeros((min(num_img, num_samples), rows, cols), dtype=np.uint8)
	y = np.zeros(min(num_lbl, num_samples), dtype=np.uint8)

	#Parse images to fill our grid
	if num_img == num_lbl:
		N = min(num_img, num_samples)
		print('Loading '+mode+' data')
		for i in tqdm(range(N)):
			x[i] = [[unpack('>B', images.read(1))[0] for c in range(cols)] for r in range(rows)]
			y[i] = unpack('>B', labels.read(1))[0]

	return x,y


def initial_weight_matrix(group_from='exc', group_to='inh', num_neurons=400, num_inputs=784):
	if group_from == 'exc' and group_to == 'inh':
		weights = np.zeros((num_neurons, num_neurons))
		for i in range(num_neurons):
			weights[i,i] = 10.4
		return weights

	if group_from == 'inh' and group_to == 'exc':
		weights = np.ones((num_neurons, num_neurons))*17
		for i in range(num_neurons):
			weights[i,i] = 0
		return weights

	if group_from == 'input' and group_to == 'exc':
		weights = 0.3*np.random.random((num_inputs, num_neurons)) + 0.01
		return weights

	return False


def compute_excitatory_neuron_classes(num_neurons, spike_monitor, input_stimulation_time, input_times, input_classes):
	neuron_spikes = {n:{c:0 for c in range(10)} for n in range(num_neurons)} #init dictionary to hold neuron spikes per class
	input_idx = 0
	for spike_idx,neuron_idx in enumerate(spike_monitor.i):
		spike_time = spike_monitor.t[spike_idx]
		while input_idx < len(input_times) and spike_time > input_times[input_idx] + input_stimulation_time:
			input_idx += 1
		if input_idx < len(input_times):
			neuron_spikes[neuron_idx][input_classes[input_idx]] += 1
		else:
			break

	neuron_classes = []
	for neuron_idx in range(num_neurons):
		spike_counts = neuron_spikes[neuron_idx]
		neuron_classes.append(max(spike_counts, key=spike_counts.get))

	return neuron_classes


def save_network(neurons, connections, neuron_classes):
	#save connections
	for connection_name,connection in connections.items():
		sparse_weights = np.array(list(zip(connection.i, connection.j, connection.w)))
		np.save('saved/'+connection_name+'_weights', sparse_weights)

	#save neuron adaptive threshold theta's
	np.save('saved/exc_theta', neurons['exc'].theta)

	#save exc neuron class labels
	np.save('saved/class_labels', np.array(neuron_classes))


def save_network_in_memory(neurons, connections, neuron_classes):
	network = {}

	#save connections
	for connection_name,connection in connections.items():
		sparse_weights = np.array(list(zip(connection.i, connection.j, connection.w)))
		network[f"{connection_name}_weights"] = sparse_weights

	#save neuron adaptive threshold theta's
	network["exc_theta"] = neurons['exc'].theta

	#save exc neuron class labels
	network["class_labels"] = np.array(neuron_classes)

	return network


def load_network(num_neurons):
	#load connections
	connection_weights = {}
	for connection_name in ['exc-inh', 'inh-exc', 'input-exc']:
		sparse_weights = np.load('saved/'+connection_name+'_weights.npy')
		if connection_name == 'input-exc':
			weight_matrix = np.zeros((784,num_neurons))
		else:
			weight_matrix = np.zeros((num_neurons,num_neurons))
		weight_matrix[np.int32(sparse_weights[:,0]), np.int32(sparse_weights[:,1])] = sparse_weights[:,2]
		connection_weights[connection_name] = weight_matrix

	#load neuron adaptive threshold theta's
	neuron_thetas = np.load('saved/exc_theta.npy')*volt

	#load exc neuron class labels
	class_labels = np.load('saved/class_labels.npy')

	return neuron_thetas,connection_weights,class_labels


def load_network_from_memory(network, num_neurons):
	#load connections
	connection_weights = {}
	for connection_name in ['exc-inh', 'inh-exc', 'input-exc']:
		sparse_weights = network[f"{connection_name}_weights"]
		if connection_name == 'input-exc':
			weight_matrix = np.zeros((784,num_neurons))
		else:
			weight_matrix = np.zeros((num_neurons,num_neurons))
		weight_matrix[np.int32(sparse_weights[:,0]), np.int32(sparse_weights[:,1])] = sparse_weights[:,2]
		connection_weights[connection_name] = weight_matrix

	#load neuron adaptive threshold theta's
	neuron_thetas = network["exc_theta"]

	#load exc neuron class labels
	class_labels = network["class_labels"]

	return neuron_thetas,connection_weights,class_labels


def calculate_predictions(num_neurons, spike_monitor, input_stimulation_time, input_times, neuron_classes):
	input_spikes = {input_idx:{c:0 for c in range(10)} for input_idx in range(len(input_times))}
	input_idx = 0
	for spike_idx,neuron_idx in enumerate(spike_monitor.i):
		spike_time = spike_monitor.t[spike_idx]
		while input_idx < len(input_times) and spike_time > input_times[input_idx] + input_stimulation_time:
			input_idx += 1
		if input_idx < len(input_times):
			input_spikes[input_idx][neuron_classes[neuron_idx]] += 1
		else:
			break

	y_pred = []
	class_counts = {c:list(neuron_classes).count(c) for c in range(10)}
	for input_idx,class_spikes in input_spikes.items():
		class_firing_rates = {c:(class_spikes[c]/class_counts[c] if class_counts[c] > 0 else 0) for c in range(10)}
		y_pred.append(max(class_firing_rates, key=class_firing_rates.get))

	return np.array(y_pred)


def apply_dropout(connection_weights, dropout_rate):
	for connection_name in connection_weights:
		nonzero_r,nonzero_c = np.nonzero(connection_weights[connection_name])
		dropout_idxs = np.random.choice(range(len(nonzero_r)), int(dropout_rate*len(nonzero_r)), replace=False)
		for idx in dropout_idxs:
			connection_weights[connection_name][nonzero_r[idx],nonzero_c[idx]] = 0
	return connection_weights


#mode = train or test, num_neurons = number of excitatory neurons / number of inhibitory neurons
def run_snn(mode='train', num_neurons=400, X_data=[], y_data=[], integration_method='euler', dt_ms=False, synapse_dropout_rate=0, params={}, saved_network=None, count_spikes=False):
	#override default simultation discrete time grid size of 100 microsec (usec)
	if dt_ms != False:
		defaultclock.dt = dt_ms*ms

	if mode == 'test':
		#neuron_thetas,saved_connection_weights,neuron_classes = load_network(num_neurons) #load the saved network state from file
		neuron_thetas,saved_connection_weights,neuron_classes = load_network_from_memory(saved_network, num_neurons)
		if synapse_dropout_rate > 0:
			saved_connection_weights = apply_dropout(saved_connection_weights, synapse_dropout_rate)

	# Define neurons and spiking behaviour
	tau_exc = params.get("tau_exc", 100)*ms #time constant for excitatory neurons
	tau_inh = params.get("tau_inh", 10)*ms #time constant for inhibitory neurons
	E_rest_exc = params.get("E_rest_exc", -65)*mV #resting membrane potential for excitatory neurons
	E_rest_inh = params.get("E_rest_inh", -60)*mV #resting membrane potential for inhibitory neurons
	E_exc_a = params.get("E_exc_a", 0)*mV #equilibrium potential for excitatory neurons
	E_inh_a = params.get("E_inh_a", -100)*mV #equilibrium potential for inhibitory neurons
	E_exc_b = params.get("E_exc_b", 0)*mV #same two constants as above but applied to inhib eqs
	E_inh_b = params.get("E_inh_b", -85)*mV

	tau_ge = params.get("tau_ge", 1)*ms #time constant of an excitatory postsynaptic potential
	tau_gi = params.get("tau_gi", 2)*ms #time constant of an inhibitory postsynaptic potential

	tau_theta = params.get("tau_theta", 1e7)*ms #time constant for adaptive membrane threshold

	#equations for excitatory neuron state variables
	excite_eqs = """
		dv/dt = ((E_rest_exc - v) + g_e*(E_exc_a - v) + g_i*(E_inh_a - v))/tau_exc : volt (unless refractory)
		dg_e/dt = -g_e/tau_ge : 1
		dg_i/dt = -g_i/tau_gi : 1
		dtheta/dt = -theta/tau_theta : volt
		dtimer/dt = 0.1 : second
	"""
	if mode == 'test':
		excite_eqs = """
			dv/dt = ((E_rest_exc - v) + g_e*(E_exc_a - v) + g_i*(E_inh_a - v))/tau_exc : volt (unless refractory)
			dg_e/dt = -g_e/tau_ge : 1
			dg_i/dt = -g_i/tau_gi : 1
			theta : volt
			dtimer/dt = 0.1 : second
		"""

	#equations for inhibitory neuron state variables
	inhib_eqs = """
		dv/dt = ((E_rest_inh - v) + g_e*(E_exc_b - v) + g_i*(E_inh_b - v))/tau_exc : volt (unless refractory)
		dg_e/dt = -g_e/tau_ge : 1
		dg_i/dt = -g_i/tau_gi : 1
	"""

	v_reset_exc = params.get("v_reset_exc", -65)*mV #reset potential for excitatory neurons
	v_reset_inh = params.get("v_reset_inh", -45)*mV #reset potential for inhibitory neurons
	v_thresh_exc = params.get("v_thresh_exc", -52-20)*mV #spike threshold potential for excitatory neurons (including offset)
	v_thresh_inh = params.get("v_thresh_inh", -40)*mV #spike threshold potential for inhibitory neurons
	refrac_exc = params.get("refrac_exc", 5)*ms #refractory period for excitatory neurons
	refrac_inh = params.get("refrac_inh", 2)*ms #refractory period for inhibitory neurons
	theta_plus_exc = params.get("theta_plus_exc", 0.05)*mV #increase amount of adaptive membrane threshold upon spike

	#threshold conditions and spike reset actions for excitatory neurons
	exc_thresh = '(v > (theta + v_thresh_exc)) and (timer > refrac_exc)'
	exc_reset = 'v = v_reset_exc; theta += theta_plus_exc; timer = 0*ms'
	if mode == 'test':
		exc_reset = 'v = v_reset_exc; timer = 0*ms'

	#threshold conditions and spike reset actions for inhibitory neurons
	inh_thresh = 'v > v_thresh_inh'
	inh_reset = 'v = v_reset_inh'

	#define excitatory and inhibitory neurons
	neurons = {}
	neurons['exc'] = NeuronGroup(num_neurons, excite_eqs, threshold=exc_thresh, refractory=refrac_exc, reset=exc_reset, method=integration_method)
	neurons['exc'].v = E_rest_exc - 40*mV
	if mode == 'train':
		neurons['exc'].theta = np.ones((num_neurons))*20*mV
	elif mode == 'test':
		neurons['exc'].theta = neuron_thetas

	neurons['inh'] = NeuronGroup(num_neurons, inhib_eqs, threshold=inh_thresh, refractory=refrac_inh, reset=inh_reset, method=integration_method)
	neurons['inh'].v = E_rest_inh - 40*mV

	#define a spike monitor for excitatory (output) neurons, to record their responses to training inputs
	monitors = {}
	monitors['spike_exc'] = SpikeMonitor(neurons['exc'])
	if count_spikes:
		monitors['spike_inh'] = SpikeMonitor(neurons['inh'])

	# Define recurrent synapses
	# w is the weight of the synapse => spike increases conductance g_e or g_i of neuron by w, which deflects neuron voltage over time
	tau_xpre = params.get("tau_xpre", 20)*ms #time constant for presynaptic trace
	nu = params.get("nu", 0.0001) #learning rate
	w_max = params.get("w_max", 1.0) #maximum synapse weight
	mu = params.get("mu", 1.0) #dependence of weight change on previous weight
	x_tar = params.get("x_tar", 1.0) #target value of the presynaptic trace at the moment of a postsynaptic spike

	#equations for STDP (synapse weight w, presynaptic trace xpre)
	stdp_eqs = """
		w : 1
		dxpre/dt = -xpre/(tau_xpre): 1 (event-driven)
	"""
	stdp_pre_actions = 'x_pre = clip(xpre+1, 0, x_tar)'
	stdp_post_actions = 'w = clip(w + nu*(xpre-x_tar)*(w_max-w)**mu, 0, w_max)'

	#create the excitatory to inhibitory synapses
	connections = {}
	if mode == 'train':
		exc_to_inh_weights = initial_weight_matrix('exc', 'inh', num_neurons)
	elif mode == 'test':
		exc_to_inh_weights = saved_connection_weights['exc-inh']

	connections['exc-inh'] = Synapses(neurons['exc'], neurons['inh'], model=stdp_eqs, on_pre='g_e_post += w; '+stdp_pre_actions, on_post=stdp_post_actions, method=integration_method)
	connections['exc-inh'].connect(j='i') #NOTE: source code had this as .connect(True)
	connections['exc-inh'].w = exc_to_inh_weights[connections['exc-inh'].i, connections['exc-inh'].j]

	#create the inhibitory to excitatory synapses
	if mode == 'train':
		inh_to_exc_weights = initial_weight_matrix('inh', 'exc', num_neurons)
	elif mode == 'test':
		inh_to_exc_weights = saved_connection_weights['inh-exc']

	connections['inh-exc'] = Synapses(neurons['inh'], neurons['exc'], model=stdp_eqs, on_pre='g_i_post += w; '+stdp_pre_actions, on_post=stdp_post_actions, method=integration_method)
	connections['inh-exc'].connect(condition='i != j') #NOTE: source code had this as .connect(True)
	connections['inh-exc'].w = inh_to_exc_weights[connections['inh-exc'].i, connections['inh-exc'].j]

	# Define input neurons and synapses
	num_inputs = 28*28 #size of each input image (one pixel => one neuron)
	neurons['input'] = PoissonGroup(num_inputs, 0*Hz) #grid of input neurons to deliver Poisson-distributed spike trains of pixel intensity to exc neurons

	if mode == 'train':
		input_to_exc_weights = initial_weight_matrix('input', 'exc', num_neurons, num_inputs)
	elif mode == 'test':
		input_to_exc_weights = saved_connection_weights['input-exc']

	connections['input-exc'] = Synapses(neurons['input'], neurons['exc'], model=stdp_eqs, on_pre='g_e_post += w; '+stdp_pre_actions, on_post=stdp_post_actions, method=integration_method)
	connections['input-exc'].connect(True) #all-to-all connections
	connections['input-exc'].w = input_to_exc_weights[connections['input-exc'].i, connections['input-exc'].j]

	#add delay to these connections
	min_delay = params.get("min_delay", 0)*ms
	max_delay = params.get("max_delay", 10)*ms
	connections['input-exc'].delay = 'min_delay + rand()*(max_delay - min_delay)'

	# Initialise the network
	network = Network()
	for component in [neurons, connections, monitors]:
		for key in component:
			network.add(component[key])
	network.run(0*second)

	# Run the network, iterating through the training data
	num_samples = y_data.size #number of images we want to train on (e.g, 10000)

	input_intensity = params.get("input_intensity", 2) #initial intensity of input (for calculating input neuron spike rates from pixel values), +1 for each input replay
	input_stimulation_time = params.get("input_stimulation_time", 350)*ms #time to run input neurons for each input image
	resting_time = params.get("resting_time", 150)*ms #time to run the network with no input between input stimulations, to reset decaying parameters
	min_input_spikes = params.get("min_input_spikes", 5) #minimum number of spikes required during input stimulation to move onto next input image
	progress_update_interval = params.get("progress_update_interval", 10) #print a progress line to the console at the end of each of these intervals
	prev_spike_count = 0 #holds the total number of spikes recorded before each stimulation
	input_times = [] #holds the start time in the simulation for the final stimulation period for each input
	current_simulation_time = 0*ms #holds the current simulation time

	j = 0
	while j < num_samples:
		#NOTE: source code normalised weights in network here
		spike_rates = X_data[j,:,:].reshape((784))*(input_intensity/8) #define the vector of Poisson spike rates for the input layer of neurons
		neurons['input'].rates = spike_rates*Hz
		network.run(input_stimulation_time) #run the network for the stimulation period (optional report='text' parameter to this fn)
		current_simulation_time += input_stimulation_time

		new_spike_count = monitors['spike_exc'].num_spikes #number of spikes recorded up until this point
		num_spikes = new_spike_count - prev_spike_count
		prev_spike_count = new_spike_count
		if num_spikes < min_input_spikes: #we need to show this same input again
			input_intensity += 1 #increase input intensity for the next run of this input
		else:
			input_times.append(current_simulation_time - input_stimulation_time) #record the start time for this final (for the current input) stimulation
			input_intensity = 2
			j += 1
			if j % progress_update_interval == 0:
				print(mode.capitalize()+'ing: '+str(j)+' of '+str(num_samples))

		neurons['input'].rates = 0*Hz #clear the input
		network.run(resting_time) #run the network with no input to reset decaying parameters
		current_simulation_time += resting_time

	# Compute and save the training results
	if count_spikes:
		return monitors['spike_exc'].num_spikes + monitors['spike_inh'].num_spikes

	if mode == 'train':
		print('Computing output classes and saving network parameters')
		neuron_classes = compute_excitatory_neuron_classes(num_neurons, monitors['spike_exc'], input_stimulation_time, input_times, y_data)
		#save_network(neurons, connections, neuron_classes)
		return save_network_in_memory(neurons, connections, neuron_classes)
	elif mode == 'test':
		y_pred = calculate_predictions(num_neurons, monitors['spike_exc'], input_stimulation_time, input_times, neuron_classes)
		return y_pred


NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES = 10
NETWORK_SIZE = 100
PARAM_SET = {"tau_exc": 100, "tau_inh": 10, "tau_ge": 1, "tau_gi": 2, "tau_theta": 1e7, "nu": 0.0001}

if __name__ == "__main__":
	X_train,y_train = load_mnist_data('train', 'mnist_data/', NUM_TRAIN_SAMPLES)
	X_test,y_test = load_mnist_data('test', 'mnist_data/', NUM_TEST_SAMPLES)

	network = run_snn('train', NETWORK_SIZE, X_train, y_train, params=PARAM_SET)
	y_pred = run_snn('test', NETWORK_SIZE, X_test, y_test, params=PARAM_SET, saved_network=network)
	accuracy = metrics.accuracy_score(y_test, y_pred)
	print(f"accuracy = {accuracy}")
