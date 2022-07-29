import math
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pmlb import fetch_data
import sklearn.metrics as metrics
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split


class Network:
	def __init__(self, params, neuron_model, neurogenesis_model, load_from=False):
		#general simulation parameters
		self.t = 0.0
		self.dt = (params['dt'] if 'dt' in params else 0.01) #network timestep size in seconds
		self.ngsis_enabled = (params['ngsis_enabled'] if 'ngsis_enabled' in params else True)
		self.supp_process_freq = (params['supp_process_freq'] if 'supp_process_freq' in params else 50)
		self.default_classification = (params['default_classification'] if 'default_classification' in params else None)
		self.float_tol = (params['float_tol'] if 'float_tol' in params else 1e3*np.finfo(float).eps) #smallest nonzero neuron/synapse property
		self.label_assign_method = (params['label_assign_method'] if 'label_assign_method' in params else "spike_count")

		#store the given neuron and neurogenesis models
		self.neuron_model = neuron_model
		self.neurogenesis_model = neurogenesis_model

		#load from file/memory
		if load_from and isinstance(load_from, str):
			self.load(load_from)
		elif load_from and isinstance(load_from, dict):
			self.load_from_memory(load_from)

	def set_neurogenesis_rate(self, r_n):
		self.neurogenesis_model.set_rate(r_n)

	def size(self):
		return {'neurons':self.neuron_model.num_neurons(), 'synapses':self.neuron_model.num_synapses()}

	def run(self, duration, freeze=False, monitor_spikes=False, ngsis_exit=False, monitor_detailed=False):
		spikes = {} if monitor_detailed else []
		num_steps = int(np.ceil(duration/self.dt))
		for step in range(num_steps):
			self.neuron_model.integrate_step(self.t, self.dt, freeze)
			if monitor_spikes:
				spikes += self.neuron_model.handle_spikes(self.t, self.dt, freeze, monitor=True)
			elif monitor_detailed: #for visualisation
				spikes[self.t] = self.neuron_model.handle_spikes(self.t, self.dt, freeze, monitor_detailed=True)
			else:
				self.neuron_model.handle_spikes(self.t, self.dt, freeze)
			if not freeze and step % self.supp_process_freq == 0:
				self.zero_out_small_numbers()
				self.neurogenesis_model.handle_cell_death(self.neuron_model, self.t)
				if self.ngsis_enabled and self.neurogenesis_model.is_time(self.t):
					if ngsis_exit:
						self.t += self.dt
						break
					self.neurogenesis_model.generate_next_occurrence_time(self.t)
					self.neurogenesis_model.neurogenesis(self.neuron_model, self.t)
				self.neuron_model.handle_maturation(self.t)
			self.t += self.dt

		if monitor_spikes or monitor_detailed:
			return spikes

	def train_with_replay(self, inputs, input_duration, cooldown_duration=0, min_spikes=5, intensity_increase=1.3, ngsis_exit=False, progress=True):
		for s,sample in enumerate(tqdm(inputs, disable = not progress)):
			sample = sample.flatten()
			satisfied = False
			intensity = 1
			while not satisfied:
				self.neuron_model.generate_input_spikes(intensity*sample, input_duration, self.t, self.dt)
				spikes = self.run(input_duration, monitor_spikes=True, ngsis_exit=ngsis_exit)
				intensity *= intensity_increase
				if len(spikes) >= min_spikes or intensity >= 10:
					satisfied = True
			if cooldown_duration > 0:
				self.run(cooldown_duration, ngsis_exit=ngsis_exit)
			if self.neurogenesis_model.per_sample_action:
				self.neurogenesis_model.finished_train_sample(self.neuron_model)
			if self.neuron_model.num_neurons() == 0:
				return False
		return True

	def label_neurons(self, input_spikes, input_labels, labels): # every label will appear at least once
		output_neuron_ids = [n for n in self.neuron_model.get_neuron_ids() if self.neuron_model.is_output_neuron(n)]

		if len(output_neuron_ids) >= len(labels): #this network can actually handle the task, otherwise add no labels to ensure no preds
			#compute a map from output neuron ID and label to total spike count
			neuron_spikes = {n:{l:0 for l in labels} for n in output_neuron_ids}
			for s,spikes in enumerate(input_spikes):
				label = input_labels[s]
				for n in spikes:
					if n in output_neuron_ids:
						neuron_spikes[n][label] += 1

			#compute a map from label to a list of neuron preferences
			label_prefs = {}
			for label in labels:
				if self.label_assign_method == "spike_count":
					label_spikes = {n: neuron_spikes[n][label] for n in output_neuron_ids}
				elif self.label_assign_method == "spike_proportion":
					label_spikes = {n: neuron_spikes[n][label]/sum(neuron_spikes[n].values()) for n in output_neuron_ids}
				label_prefs[label] = sorted(label_spikes, key=label_spikes.get, reverse=True)

			#assign each label to its top preference neuron
			assigned_neurons = []
			unassigned_labels = labels.copy()
			while len(unassigned_labels) > 0:
				label = unassigned_labels[0]
				top_pref = label_prefs[label][0]
				other_top_prefs = {l:label_prefs[l][0] for l in unassigned_labels if l != label}
				if top_pref not in other_top_prefs.values(): #label's top preference neuron is unique
					self.neuron_model.assign_label(top_pref, label)
					unassigned_labels.remove(label)
				else:  #label's top preference neuron is not unique so take the neuron's favourite choice
					n = top_pref
					pref_labels = [label] + [l for l in other_top_prefs if other_top_prefs[l] == n]
					pref_label_spikes = {l:neuron_spikes[n][l] for l in pref_labels}
					new_label = max(pref_label_spikes, key=pref_label_spikes.get)
					self.neuron_model.assign_label(n, new_label)
					unassigned_labels.remove(new_label)

				assigned_neurons.append(top_pref)
				for l in label_prefs:
					if top_pref in label_prefs[l]:
						label_prefs[l].remove(top_pref)

			#assign remaining output neurons to their highest spiking label
			for n in neuron_spikes:
				if n not in assigned_neurons:
					if sum(neuron_spikes[n].values()) > 0:
						self.neuron_model.assign_label(n, max(neuron_spikes[n], key=neuron_spikes[n].get))
					else:
						self.neuron_model.clear_label(n)

	def record_spikes_for_input_replay(self, sample, input_duration, min_spikes=5, intensity_increase=1.3):
		satisfied = False
		intensity = 1
		while not satisfied:
			self.neuron_model.generate_input_spikes(intensity*sample, input_duration, self.t, self.dt)
			spikes = self.run(input_duration, True, True)
			intensity *= intensity_increase
			if len(spikes) >= min_spikes or intensity >= 10:
				satisfied = True
		return spikes

	def label_with_replay(self, inputs, input_duration, input_labels, labels, min_spikes=5, intensity_increase=1.3, progress=True):
		spikes = []
		for sample in tqdm(inputs, disable = not progress):
			sample = sample.flatten()
			spikes.append(self.record_spikes_for_input_replay(sample, input_duration, min_spikes=5, intensity_increase=1.3))
		self.label_neurons(spikes, input_labels, labels)
		self.reset_after_freeze()

	def classify(self, spikes):
		labels = [self.neuron_model.get_label(n) for n in spikes if n != None and self.neuron_model.is_output_neuron(n) and self.neuron_model.has_label(n)]
		if len(labels) > 0:
			return max(labels, key=labels.count)
		return self.default_classification

	def test_with_replay(self, inputs, input_duration, min_spikes=5, intensity_increase=1.3, progress=True):
		preds = []
		for sample in tqdm(inputs, disable = not progress):
			sample = sample.flatten()
			satisfied = False
			intensity = 1
			while not satisfied:
				self.neuron_model.generate_input_spikes(intensity*sample, input_duration, self.t, self.dt)
				spikes = self.run(input_duration, True, True)
				intensity *= intensity_increase
				if len(spikes) >= min_spikes or intensity >= 10:
					satisfied = True
			preds.append(self.classify(spikes))
		self.reset_after_freeze()
		return preds

	def reset_after_freeze(self):
		self.neurogenesis_model.generate_next_occurrence_time(self.t)
		self.neurogenesis_model.reset_after_freeze(self.neuron_model,self.t)

	def save(self, file_name):
		network = {'t':self.t, 'neuron_model':self.neuron_model.save(), 'neurogenesis_model':self.neurogenesis_model.save()}
		with open(file_name+'.pkl', 'wb') as f:
			pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

	def load(self, file_name):
		network = {}
		with open(file_name+'.pkl', 'rb') as f:
			network = pickle.load(f)

		self.t = network['t']
		self.neuron_model.load(network['neuron_model'], self.dt)
		self.neurogenesis_model.load(network['neurogenesis_model'])

	def load_from_memory(self, network):
		self.t = network['t']
		self.neuron_model.load(network['neuron_model'], self.dt)
		self.neurogenesis_model.load(network['neurogenesis_model'])

	def reset_time(self):
		self.t = 0
		self.neurogenesis_model.generate_next_occurrence_time(self.t)

	def visualise(self, file_name=None, max_edge_width=5):
		import networkx as nx
		import matplotlib.pyplot as plt

		G = nx.DiGraph()

		neuron_coords = self.neuron_model.get_neuron_coords()
		synapses_info = self.neuron_model.get_synapses_info()

		out_edges = {}
		for neuron in neuron_coords:
			G.add_node(neuron['id'], pos=(neuron['x'],neuron['y']), color='#3675c2')
			out_edges[neuron['id']] = []
		positions = nx.get_node_attributes(G, 'pos')
		node_colours = [G.nodes[neuron['id']]['color'] for neuron in neuron_coords]

		edge_colours = []
		edge_widths = []
		for synapse in synapses_info:
			G.add_edge(synapse['from'], synapse['to'], weight=abs(synapse['weight']))
			out_edges[synapse['from']].append(len(edge_colours))
			edge_colours.append(get_synapse_colour(0.5))
			edge_widths.append(max_edge_width*synapse['weight'])

		fig,ax = plt.subplots()
		nx.draw(G, positions, node_color=node_colours, edge_color=edge_colours, width=edge_widths, ax=ax, arrows=False)

		if file_name:
			plt.savefig(file_name)
		else:
			plt.show()
		plt.close('all')

	def zero_out_small_numbers(self):
		neuron_props = self.neuron_model.neuron_float_props + self.neurogenesis_model.float_props
		for n in self.neuron_model.get_neuron_ids():
			for prop in neuron_props:
				if abs(self.neuron_model.get_property(n, prop)) <= self.float_tol:
					self.neuron_model.set_property(n, prop, 0)

		synapse_props = self.neuron_model.synapse_float_props
		for s in self.neuron_model.get_synapse_ids():
			for prop in synapse_props:
				if abs(self.neuron_model.get_synapse_property(s, prop)) <= self.float_tol:
					self.neuron_model.set_synapse_property(s, prop, 0)

	def update_neuron_params(self, params):
		self.neuron_model.update_params(params)

	def update_neurogenesis_params(self, params):
		self.neurogenesis_model.update_params(params)

	def utilisation(self, w_min=0.1):
		return self.neuron_model.synapse_utilisation(w_min)

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			setattr(result, k, deepcopy(v, memo))
		return result


def round_down_to_nearest_formatted(x, dx):
	x = math.floor(x/dx)*dx
	return '{:.3f}'.format(x)


class Neuron:
	def __init__(self, params, neurogenesis_model):
		self.tau_u_exc = (params['tau_u_exc'] if 'tau_u_exc' in params else 0.1) #time constant of membrane potential u for excitatory neurons
		self.tau_u_inh = (params['tau_u_inh'] if 'tau_u_inh' in params else 0.1) #time constant of membrane potential u for inhibitory neurons
		self.u_rest_exc = (params['u_rest_exc'] if 'u_rest_exc' in params else -65) #resting membrane potential for excitatory neurons
		self.u_rest_inh = (params['u_rest_inh'] if 'u_rest_inh' in params else -65) #resting membrane potential for inhibitory neurons
		self.u_exc_exc = (params['u_exc_exc'] if 'u_exc_exc' in params else 0) #excitation target membrane potential in excitatory neurons
		self.u_inh_exc = (params['u_inh_exc'] if 'u_inh_exc' in params else -100) #inhibition target membrane potential in excitatory neurons
		self.u_exc_inh = (params['u_exc_inh'] if 'u_exc_inh' in params else 0) #excitation target membrane potential in inhibitory neurons
		self.u_inh_inh = (params['u_inh_inh'] if 'u_inh_inh' in params else -100) #inhibition target membrane potential in inhibitory neurons
		self.tau_ge = (params['tau_ge'] if 'tau_ge' in params else 0.01) #time constant of excitatory conductance (postsynaptic potential) g_e
		self.tau_gi = (params['tau_gi'] if 'tau_gi' in params else 0.01) #time constant of inhibitory conductance (postsynaptic potential) g_i
		self.tau_theta = (params['tau_theta'] if 'tau_theta' in params else 0.1) #time constant of adaptive membrane threshold theta
		self.u_reset_exc = (params['u_reset_exc'] if 'u_reset_exc' in params else -65) #post-spike reset membrane potential for excitatory neurons
		self.u_reset_inh = (params['u_reset_inh'] if 'u_reset_inh' in params else -65) #post-spike reset membrane potential for inhibitory neurons
		self.u_spike_exc = (params['u_spike_exc'] if 'u_spike_exc' in params else -50) #spiking threshold of membrane potential for excitatory neurons
		self.u_spike_inh = (params['u_spike_inh'] if 'u_spike_inh' in params else -50) #spiking threshold of membrane potential for inhibitory neurons
		self.refrac_exc = (params['refrac_exc'] if 'refrac_exc' in params else 0.005) #refractory period for excitatory neurons
		self.refrac_inh = (params['refrac_inh'] if 'refrac_inh' in params else 0.005) #refractory period for inhibitory neurons
		self.theta_0 = (params['theta_0'] if 'theta_0' in params else 0) #initial value of adaptive membrane threshold theta
		self.theta_plus_exc = (params['theta_plus_exc'] if 'theta_plus_exc' in params else 5) #increase amount of adaptive membrane threshold theta upon spike
		self.tau_x = (params['tau_x'] if 'tau_x' in params else 0.02) #time constant of presynaptic trace x
		self.nu = (params['nu'] if 'nu' in params else 0.0001) #STDP learning rate
		self.w_max = (params['w_max'] if 'w_max' in params else 1) #maximum synapse weight
		self.mu = (params['mu'] if 'mu' in params else 1) #dependence of synapse weight change on previous weight
		self.x_tar = (params['x_tar'] if 'x_tar' in params else 0.8) #target value of the presynaptic trace at the moment of a postsynaptic spike
		self.dg = (params['dg'] if 'dg' in params else 0.01) #size of g_e and g_i grids for precomputing u,g tables
		self.du = (params['du'] if 'du' in params else 0.01) #size of u grid for precomputing u,g tables
		self.output_x_range = (params['output_x_range'] if 'output_x_range' in params else [0,1]) #inclusive x-axis range of output neurons
		self.w_min = (params['w_min'] if 'w_min' in params else 0.05) #minimum absolute synapse weight under which synapses are pruned
		self.neuron_float_props = (params['neuron_float_props'] if 'neuron_float_props' in params else ['u','g_e','g_i']) #integrated floats in neurons
		self.synapse_float_props = (params['synapse_float_props'] if 'synapse_float_props' in params else ['x','w']) #integrated floats in synapses

		self.input_neurons = {}
		self.neurons = {}
		self.synapses = {}
		self.neurogenesis_model = neurogenesis_model
		self.input_spikes = {}
		self.spikes = []
		self.ug_exc_table = {}
		self.ug_inh_table = {}
		self.count_spikes = False
		self.spike_count = {"input":0, "other":0}

	def update_params(self, params):
		for p in params:
			setattr(self, p, params[p])

	def create_synapse(self, n1, n2, w=None, other_props={}):
		if w == None:
			w = np.random.uniform(0,1) #synapse weights always positive
		s = (max(list(self.synapses))+1 if self.synapses else 0)
		self.synapses[s] = {'w':w, 'x':0, 't0':0, 'pre':n1, 'post':n2, **other_props} #t0 is last time we updated x_pre
		if isinstance(n1, str):
			self.input_neurons[n1]['out_synapses'].append(s)
		else:
			self.neurons[n1]['out_synapses'].append(s)
		self.neurons[n2]['in_synapses'].append(s)

	def num_neurons(self):
		return len(self.neurons)

	def get_neuron_ids(self):
		return list(self.neurons.keys())

	def get_synapse_ids(self):
		return list(self.synapses.keys())

	def num_synapses(self):
		return len(self.synapses)

	def get_neurons(self):
		return list(self.neurons.values())

	def get_neuron(self, n):
		return self.neurons[n]

	def generate_poisson_spikes(self, firing_rate, num_steps, t, dt):
		x = np.random.uniform(0,1,num_steps)
		spike_steps = np.argwhere(x <= firing_rate*dt).flatten()
		return [round_down_to_nearest_formatted(t + s*dt, dt) for s in spike_steps]

	def generate_input_spikes(self, input_rates, input_duration, t, dt):
		self.input_spikes = {}
		num_steps = int(np.ceil(input_duration/dt))
		for num,n in enumerate(self.input_neurons.keys()):
			spike_times = self.generate_poisson_spikes(input_rates[num], num_steps, t, dt)
			for spike_time in spike_times:
				if spike_time not in self.input_spikes:
					self.input_spikes[spike_time] = []
				self.input_spikes[spike_time].append(n)

	def neuron_params_exc(self):
		return (self.tau_u_exc, self.u_rest_exc, self.u_exc_exc, self.u_inh_exc, self.tau_ge, self.tau_gi)

	def neuron_params_inh(self):
		return (self.tau_u_inh, self.u_rest_inh, self.u_exc_inh, self.u_inh_inh, self.tau_ge, self.tau_gi)

	@staticmethod
	def neuron_gradients(variables, t, tau_u, u_rest, u_exc, u_inh, tau_ge, tau_gi):
		dudt = (u_rest-variables[0] + variables[1]*(u_exc-variables[0]) + variables[2]*(u_inh-variables[0]))/tau_u
		dgedt = -variables[1]/tau_ge
		dgidt = -variables[2]/tau_gi
		return [dudt,dgedt,dgidt]

	def integrate_step(self, t, dt, freeze=False):
		self.spikes = []
		for n in self.neurons:
			neuron = self.neurons[n]

			params = self.neuron_params_exc() if neuron['type'] == 'exc' else self.neuron_params_inh()
			[neuron['u'],neuron['g_e'],neuron['g_i']] = odeint(self.neuron_gradients, [neuron['u'],neuron['g_e'],neuron['g_i']], [t, t+dt], args=params)[-1]

			if not freeze and neuron['type'] == 'exc':
				neuron['theta'] = neuron['theta']*np.exp(-dt/self.tau_theta)

			if (
				(neuron['type'] == 'exc' and neuron['u'] >= self.u_spike_exc + neuron['theta'] and neuron['t_spike'] < t+self.refrac_exc) or
				(neuron['type'] == 'inh' and neuron['u'] >= self.u_spike_inh and neuron['t_spike'] < t+self.refrac_inh)
			):
				self.spikes.append(n)
				neuron['t_spike'] = t

	def update_synapse_trace(self, s, t):
		self.synapses[s]['x'] = self.synapses[s]['x']*np.exp(-(t-self.synapses[s]['t0'])/self.tau_x)
		self.synapses[s]['t0'] = t

	def handle_spikes(self, t, dt, freeze=False, monitor=False, monitor_detailed=False):
		t_formatted = round_down_to_nearest_formatted(t,dt)
		if t_formatted in self.input_spikes:
			for n in self.input_spikes[t_formatted]:
				neuron = self.input_neurons[n]

				#for any outgoing synapses, pre-synaptic trace x += 1
				for s in neuron['out_synapses']:
					self.update_synapse_trace(s,t)
					self.synapses[s]['x'] += 1

				#for any outgoing connected neurons, their exc postsynaptic potential (conductance) g_e += w (synapse weight)
				for s in neuron['out_synapses']:
					post_neuron = self.neurons[self.synapses[s]['post']]
					post_neuron['g_e'] += self.neurogenesis_model.enhance_spike(self.synapses[s]['w']) if post_neuron['enhanced'] else self.synapses[s]['w']

		prune_synapses = []
		for n in self.spikes:
			neuron = self.neurons[n]

			#reset the membrane potential u
			neuron['u'] = (self.u_reset_exc if neuron['type'] == 'exc' else self.u_reset_inh)

			#if excitatory neuron, increment the adaptive threshold theta to make it slightly harder to spike again
			if neuron['type'] == 'exc' and not freeze:
				neuron['theta'] += self.theta_plus_exc

			#let the neurogenesis model handle the spike, to update any relevant neuron properties
			self.neurogenesis_model.handle_spike(neuron, t)

			#for any outgoing synapses, pre-synaptic trace x += 1
			for s in neuron['out_synapses']:
				self.update_synapse_trace(s,t)
				self.synapses[s]['x'] += 1

			#for any outgoing connected neurons, their exc/inh postsynaptic potential (conductance) g += w (synapse weight)
			for s in neuron['out_synapses']:
				post_neuron = self.neurons[self.synapses[s]['post']]
				target_g = ('g_e' if neuron['type'] == 'exc' else 'g_i')
				post_neuron[target_g] += self.neurogenesis_model.enhance_spike(self.synapses[s]['w']) if post_neuron['enhanced'] else self.synapses[s]['w']

			#for any incoming synapses, apply STDP to synapse weight w
			if not freeze:
				for s in neuron['in_synapses']:
					if s in self.synapses:
						synapse = self.synapses[s]
						synapse_enhanced = (neuron['enhanced'] or (self.synapses[s]['pre'] in self.neurons and self.neurons[self.synapses[s]['pre']]['enhanced']))
						learning_rate = self.neurogenesis_model.enhance_learning_rate(self.nu) if synapse_enhanced else self.nu
						self.update_synapse_trace(s,t)
						synapse['w'] += learning_rate*(synapse['x'] - self.x_tar)*((self.w_max - synapse['w'])*synapse['w'])**self.mu

						if synapse['w'] <= self.w_min:
							prune_synapses.append(s)

		#prune any weak synapses
		if not freeze:
			for s in prune_synapses:
				if s in self.synapses:
					m = self.synapses[s]['pre']
					if isinstance(m, str):
						self.input_neurons[m]['out_synapses'].remove(s)
					else:
						self.neurons[m]['out_synapses'].remove(s)

					n = self.synapses[s]['post']
					self.neurons[n]['in_synapses'].remove(s)

					del self.synapses[s]

		if self.count_spikes:
			self.spike_count["input"] += len(self.input_spikes[t_formatted] if t_formatted in self.input_spikes else [])
			self.spike_count["other"] += len(self.spikes)

		if monitor:
			return self.spikes
		elif monitor_detailed:
			input_spikes = self.input_spikes[t_formatted] if t_formatted in self.input_spikes else []
			return input_spikes + self.spikes

	def create_neuron(self, x, y, z, neuron_type, other_props, enhanced=False, enhanced_until=0):
		n = int(max(list(self.neurons.keys()))+1 if self.neurons else 0)
		if neuron_type == 'exc':
			self.neurons[n] = {'id':n, 'x':x, 'y':y, 'z':z, 'type':'exc', 'u':self.u_rest_exc, 'g_e':0, 'g_i':0, 'theta':self.theta_0, 't_spike':-1, **other_props, 'in_synapses':[], 'out_synapses':[], 'enhanced':enhanced, 'enh_until':enhanced_until, 'label':None}
		else: #inh neurons don't have an adaptive threshold theta
			self.neurons[n] = {'id':n, 'x':x, 'y':y, 'z':z, 'type':'inh', 'u':self.u_rest_inh, 'g_e':0, 'g_i':0, 't_spike':-1, **other_props, 'in_synapses':[], 'out_synapses':[], 'enhanced':enhanced, 'enh_until':enhanced_until, 'label':None}
		return n

	def destroy_neuron(self, n):
		neuron = self.neurons[n]
		for s in neuron['in_synapses']:
			if s in self.synapses:
				m = self.synapses[s]['pre']
				if isinstance(m, str):
					self.input_neurons[m]['out_synapses'].remove(s)
				else:
					self.neurons[m]['out_synapses'].remove(s)
				del self.synapses[s]
		for s in neuron['out_synapses']:
			if s in self.synapses:
				self.neurons[self.synapses[s]['post']]['in_synapses'].remove(s)
				del self.synapses[s]
		del self.neurons[n]

	def handle_maturation(self, t):
		for n in self.neurons:
			if self.neurons[n]['enhanced'] and self.neurons[n]['enh_until'] < t:
				self.neurons[n]['enhanced'] = False
				del self.neurons[n]['enh_until']

	def order_neurons_by_proximity(self, n):
		distances = {m:np.sqrt((self.neurons[m]['x']-self.neurons[n]['x'])**2 + (self.neurons[m]['y']-self.neurons[n]['y'])**2  + (self.neurons[m]['z']-self.neurons[n]['z'])**2) for m in self.neurons if m != n}
		return [(m,dist) for m,dist in sorted(distances.items(), key=lambda item: item[1])]

	def assign_label(self, neuron_id, label):
		self.neurons[neuron_id]['label'] = label

	def clear_label(self, neuron_id):
		self.neurons[neuron_id]['label'] = None

	def has_label(self, neuron_id):
		return ('label' in self.neurons[neuron_id] and self.neurons[neuron_id]['label'] != None)

	def get_label(self, neuron_id):
		return self.neurons[neuron_id]['label']

	def has_property(self, neuron_id, prop):
		if neuron_id in self.neurons:
			return prop in self.neurons[neuron_id]
		elif neuron_id in self.input_neurons:
			return prop in self.input_neurons[neuron_id]
		return False

	def get_property(self, neuron_id, prop):
		if neuron_id in self.neurons:
			return self.neurons[neuron_id][prop]
		elif neuron_id in self.input_neurons:
			return self.input_neurons[neuron_id][prop]
		return None

	def set_property(self, neuron_id, prop, value):
		if neuron_id == 'all':
			for n in self.neurons:
				self.neurons[n][prop] = value
		else:
			self.neurons[neuron_id][prop] = value

	def does_synapse_exist(self, synapse_id):
		return synapse_id in self.synapses

	def get_synapse_property(self, synapse_id, prop):
		return self.synapses[synapse_id][prop]

	def set_synapse_property(self, synapse_id, prop, value):
		self.synapses[synapse_id][prop] = value

	def prune_synapse(self, synapse_id):
		m = self.synapses[synapse_id]['pre']
		if isinstance(m, str):
			self.input_neurons[m]['out_synapses'].remove(synapse_id)
		else:
			self.neurons[m]['out_synapses'].remove(synapse_id)

		n = self.synapses[synapse_id]['post']
		self.neurons[n]['in_synapses'].remove(synapse_id)

		del self.synapses[synapse_id]

	def get_neuron_coords(self):
		coords = []
		for n in self.input_neurons:
			neuron = self.input_neurons[n]
			coords.append({'id':n, 'x':neuron['x'], 'y':neuron['y'], 'z':neuron['z']})
		for n in self.neurons:
			neuron = self.neurons[n]
			coords.append({'id':n, 'x':neuron['x'], 'y':neuron['y'], 'z':neuron['z']})
		return coords

	def get_hidden_neuron_properties(self, props):
		info = []
		for n in self.neurons:
			neuron = self.neurons[n]
			neuron_info = {p:neuron[p] for p in props if p in neuron}
			neuron_info['id'] = n
			info.append(neuron_info)
		return info

	def is_output_neuron(self, neuron_id):
		return (self.neurons[neuron_id]['x'] >= self.output_x_range[0] and self.neurons[neuron_id]['x'] <= self.output_x_range[1])

	def get_synapses_info(self):
		synapses_info = []
		for s in self.synapses:
			synapse = self.synapses[s]
			synapses_info.append({'id':s, 'from':synapse['pre'], 'to':synapse['post'], 'weight':abs(synapse['w'])})
		return synapses_info

	def save(self):
		return {'input_neurons':self.input_neurons, 'neurons':self.neurons, 'synapses':self.synapses, 'ug_exc_table':self.ug_exc_table, 'ug_inh_table':self.ug_inh_table}

	def load(self, data, dt):
		self.input_neurons = data['input_neurons']
		self.neurons = data['neurons']
		self.synapses = data['synapses']

	def weights_snapshot(self):
		weights = {}
		for s in self.synapses:
			weights[s] = self.synapses[s]['w']
		return weights

	def neurons_snapshot(self, prop):
		if prop == 'u_spikes':
			return {'spikes': self.spikes, **{n: self.neurons[n]['u'] for n in self.neurons}}
		return {n: self.neurons[n][prop] for n in self.neurons}

	def synapse_dropout(self, dropout_rate=0.5):
		synapse_ids = list(self.synapses.keys())
		dropout_synapse_ids = random.sample(synapse_ids, k=int(dropout_rate*len(synapse_ids)))
		for s in dropout_synapse_ids:
			m = self.synapses[s]['pre']
			if isinstance(m, str):
				self.input_neurons[m]['out_synapses'].remove(s)
			else:
				self.neurons[m]['out_synapses'].remove(s)

			n = self.synapses[s]['post']
			self.neurons[n]['in_synapses'].remove(s)

			del self.synapses[s]

	def synapse_utilisation(self, w_min=0.1):
		if len(self.synapses) == 0:
			return 0

		small_weights = 0
		for s in self.synapses:
			if self.synapses[s]["w"] < w_min:
				small_weights += 1
		return 1 - small_weights/len(self.synapses)

	def set_count_spikes(self, count_spikes=False):
		self.count_spikes = count_spikes
		self.spike_count = {"input":0, "other":0}

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			setattr(result, k, deepcopy(v, memo))
		return result


class Neurogenesis:
	def __init__(self, params):
		self.n_s = (params['n_s'] if 'n_s' in params else 0.1) #neuromodulator discrete increase on spike
		self.tau_n = (params['tau_n'] if 'tau_n' in params else 5) #time constant of neuromodulator variable (larger means slower decay)
		self.n_d = (params['n_d'] if 'n_d' in params else 0.05) #neuromodulator cell death threshold
		self.r_n = (params['r_n'] if 'r_n' in params else 1) #neurogenesis cell production rate (num cells per second)
		self.t_max = (params['t_max'] if 't_max' in params else None) #if not none, time after which neurogenesis stops
		self.sigma_r = (params['sigma_r'] if 'sigma_r' in params else 0.25) #standard deviation of neurogenesis timing noise
		self.g_enh = (params['g_enh'] if 'g_enh' in params else 1) #newborn neurons enhanced excitability factor
		self.nu_enh = (params['nu_enh'] if 'nu_enh' in params else 1) #newborn neurons enhanced STDP factor
		self.T_enh = (params['T_enh'] if 'T_enh' in params else 0) #newborn neurons enhanced period length (in seconds)
		self.s_n = (params['s_n'] if 's_n' in params else 1) #new synapse neuromodulator dependency power law parameter
		self.s_d = (params['s_d'] if 's_d' in params else 1) #new synapse distance dependency exponential parameter
		self.s_max = (params['s_max'] if 's_max' in params else 10) #max synapses for a newborn neuron
		self.sigma_n = (params['sigma_n'] if 'sigma_n' in params else 0) #standard deviation of neurogenesis cell position noise
		self.neuron_dims = (params['neuron_dims'] if 'neuron_dims' in params else 3) #number of coordinates for neuron position
		self.method = (params['method'] if 'method' in params else 'active') #method for computing position of new neuron in neurogenesis
		self.synapse_dirn = (params['synapse_dirn'] if 'synapse_dirn' in params else 'forward') #method for determining new synapse dirn
		self.float_props = (params['float_props'] if 'float_props' in params else ['n']) #integrated floats in neurons
		self.per_sample_action = False

		self.next_ng_time = 0

	def update_params(self, params):
		for p in params:
			setattr(self, p, params[p])

	def set_rate(self, r_n):
		self.r_n = r_n

	def generate_next_occurrence_time(self, t):
		self.next_ng_time = t + max(0.0,np.random.normal(1,self.sigma_r))*(1/self.r_n)
		if self.t_max and self.next_ng_time > self.t_max:
			self.next_ng_time = -1 # disabled neurogenesis

	def is_time(self, t):
		return (self.next_ng_time >= 0 and t >= self.next_ng_time)

	def initial_neuron_props(self):
		return {'n':0.5,'t0':0}

	def update_neuromodulator_density(self, neuron, t):
		neuron['n'] = neuron['n']*np.exp(-(t-neuron['t0'])/self.tau_n)
		neuron['t0'] = t

	def update_neuromodulator_densities(self, neuron_model, t):
		for neuron in neuron_model.get_neurons():
			self.update_neuromodulator_density(neuron,t)

	def handle_spike(self, neuron, t):
		#increase the neuron's neuromodulator variable n
		self.update_neuromodulator_density(neuron, t)
		neuron['n'] = min(neuron['n'] + self.n_s, 1)

	def handle_cell_death(self, neuron_model, t):
		self.update_neuromodulator_densities(neuron_model,t)
		neurons = neuron_model.get_hidden_neuron_properties(['n'])
		for neuron in neurons:
			if neuron['n'] < self.n_d and ('eternal' not in neuron or not neuron['eternal']):
				neuron_model.destroy_neuron(neuron['id'])

	def position_neuron(self, neuron_model):
		neurons = neuron_model.get_hidden_neuron_properties(['n', 'x', 'y', 'z'])

		if self.method == 'active':
			N = np.array([neuron['n'] for neuron in neurons])
		elif self.method == 'active_square':
			N = np.array([neuron['n']**2 for neuron in neurons])
		elif self.method == 'inactive':
			N = np.array([1-neuron['n'] for neuron in neurons])
		elif self.method == 'inactive_square':
			N = np.array([(1-neuron['n'])**2 for neuron in neurons])

		if self.sigma_n > 0:
			N_noise = np.random.lognormal(0,self.sigma_n,N.size)
			N_noise[N_noise < 0] = 0
			N *= N_noise
		N_norm = np.linalg.norm(N,1)
		if N_norm > 0:
			N /= N_norm

		X = np.array([neuron['x'] for neuron in neurons])
		x = np.dot(N,X)

		Y = np.array([neuron['y'] for neuron in neurons])
		y = np.dot(N,Y)

		if self.neuron_dims == 3:
			Z = np.array([neuron['z'] for neuron in neurons])
			z = np.dot(N,Z)
			return {'x':x, 'y':y, 'z':z}
		else:
			return {'x':x, 'y':y}

	def neurogenesis(self, neuron_model, t):
		self.update_neuromodulator_densities(neuron_model,t)

		# use particular ngsis scheme to determine position of new neuron
		new_pos = self.position_neuron(neuron_model)

		# create new neuron
		if self.neuron_dims == 3:
			n = neuron_model.create_neuron(new_pos['x'], new_pos['y'], new_pos['z'], 'exc', {'n':0.5,'t0':t}, True, t+self.T_enh)
		else:
			n = neuron_model.create_neuron(new_pos['x'], new_pos['y'], {'n':0.5,'t0':t}, True, t+self.T_enh)

		# create synapses for new neuron
		num_synapses = 0
		local_types = {'exc':0, 'inh':0}

		# create synapses by iterating through local nbhd of new neuron
		for (m,dist) in neuron_model.order_neurons_by_proximity(n):
			self.update_neuromodulator_density(neuron_model.get_neuron(m), t)
			P_synapse = (neuron_model.get_property(m,'n')**self.s_n)*np.exp(-self.s_d*dist)
			if np.random.uniform(0,1) <= P_synapse:
				if (
					(self.synapse_dirn == "random" and np.random.uniform(0,1) <= 0.5) or
					(self.synapse_dirn == "forward" and neuron_model.get_property(m, 'x') <= new_pos['x'])
				):
					neuron_model.create_synapse(m, n)
				else:
					neuron_model.create_synapse(n, m)
				num_synapses += 1

			if neuron_model.has_property(m,'type'):
				local_types[neuron_model.get_property(m,'type')] += 1

			if num_synapses == self.s_max:
				break

		if local_types['exc'] > local_types['inh']:
			neuron_model.set_property(n, 'type', 'exc')
		elif local_types['inh'] > local_types['exc']:
			neuron_model.set_property(n, 'type', 'inh')

	def enhance_learning_rate(self, nu):
		return self.nu_enh*nu

	def enhance_spike(self, w):
		return self.g_enh*w

	def reset_after_freeze(self, neuron_model, t):
		neuron_model.set_property('all', 'n', 0.5)
		neuron_model.set_property('all', 't0', t)

	def save(self):
		return {'next_ng_time':self.next_ng_time}

	def load(self, data):
		self.next_ng_time = data['next_ng_time']

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			setattr(result, k, deepcopy(v, memo))
		return result


def get_spiking_network(input_size, midlayer_size, output_size, neuron_params, ngsis_params):
	#define the network structure
	network = {
		't': 0,
		'neuron_model': {
			'input_neurons': {'i'+str(n):{'x':0, 'y':round(n/(input_size-1),2), 'z':0, 'out_synapses':[]} for n in range(input_size)},
			'neurons': {},
			'synapses': {}
		},
		'neurogenesis_model': {'next_ng_time':0}
	}

	#initialise the neurogenesis model
	params = {
		"g_enh": 1.5,
		"nu_enh": 1.5,
		**ngsis_params
	}
	ngsis_model = Neurogenesis(params)

	#initialise the neuron model
	params = {
		'output_x_range': [0.75,1],
		'w_min': 0.1,
		'x_tar': 1,
		**neuron_params
	}
	neuron_model = Neuron(params, ngsis_model)

	#initialise the network
	params = {
		'dt': 0.001,
		'ngsis_enabled': True,
		'default_classification': None
	}
	network = Network(params, neuron_model, ngsis_model, load_from=network)

	#define the middle layer of neurons
	midlayer_ids = []
	for n in range(midlayer_size):
		midlayer_ids.append(neuron_model.create_neuron(0.5, round(n/(midlayer_size-1),2), 0, 'exc', ngsis_model.initial_neuron_props()))

	#create the output layer of neurons
	output_ids = []
	for n in range(output_size):
		initial_output_props = {'eternal':True, **ngsis_model.initial_neuron_props()}
		output_ids.append(neuron_model.create_neuron(1, round(n/(output_size-1),2), 0, 'exc', initial_output_props))

	#define the synapses for full connectivity between input and middle layers
	for n in range(input_size):
		for m in midlayer_ids:
			w = random.uniform(0,1)
			neuron_model.create_synapse('i'+str(n), m, w)

	#define the synapses for full connectivity between middle and output layers
	for n in midlayer_ids:
		for m in output_ids:
			w = random.uniform(0,1)
			neuron_model.create_synapse(n, m, w)

	return network


INPUT_MAX_FIRING_RATE_HZ = 63.75

def load_dataset(convert_to_firing_rate=False, balance_classes=False):
	X,y = fetch_data('magic', return_X_y=True)
	X /= X.max(axis=0) #normalise so each feature is in [0,1]
	if convert_to_firing_rate:
		X = INPUT_MAX_FIRING_RATE_HZ*X
	if balance_classes:
		X_0 = X[y == 0,:]
		y_0 = y[y == 0]
		X_1 = X[y == 1,:]
		y_1 = y[y == 1]
		if y_0.size > y_1.size:
			idx = np.random.choice(y_0.size, size=y_1.size, replace=False)
			y_0 = y_0[idx]
			X_0 = X_0[idx,:]
		elif y_1.size > y_0.size:
			idx = np.random.choice(y_1.size, size=y_0.size, replace=False)
			y_1 = y_1[idx]
			X_1 = X_1[idx,:]
		X = np.concatenate((X_0,X_1))
		y = np.concatenate((y_0,y_1))
	return X,y


def handle_no_predictions(y_pred, y_true):
	for s,pred in enumerate(y_pred):
		if pred is None:
			y_pred[s] = int(not bool(y_true[s]))
	return y_pred


NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES = 10
INPUT_DURATION = 0.35
COOLDOWN_DURATION = 0.15
CLASS_LABELS = [0,1]

NEURON_PARAM_SET = {'tau_u_exc': 0.001, 'tau_ge': 0.01, 'tau_theta': 0.001, 'tau_x': 0.01, 'nu': 0.001, 'tau_u_inh': 0.001, 'tau_gi': 0.01}
NGSIS_PARAM_SET = {'tau_n': 0.0001, 'n_d': -1, 'r_n': 0.05, 't_max': 75, 's_d': 1.5, 's_max': 4, 'method': 'active'}


if __name__ == "__main__":
	X,y = load_dataset(convert_to_firing_rate=True, balance_classes=True)
	X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=NUM_TRAIN_SAMPLES, test_size=NUM_TEST_SAMPLES, shuffle=True)

	network = get_spiking_network(input_size=10, midlayer_size=6, output_size=2, neuron_params=NEURON_PARAM_SET, ngsis_params=NGSIS_PARAM_SET)

	network.train_with_replay(X_train, INPUT_DURATION, COOLDOWN_DURATION, progress=True)
	network.label_with_replay(X_train, INPUT_DURATION, y_train, CLASS_LABELS, progress=True)
	y_pred = network.test_with_replay(X_test, INPUT_DURATION, progress=True)
	y_pred = handle_no_predictions(y_pred, y_test)
	accuracy = metrics.accuracy_score(y_test, y_pred)
	print(f"accuracy = {accuracy}")
