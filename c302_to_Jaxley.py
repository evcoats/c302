import sys, argparse
import numpy as np
import matplotlib.pyplot as plt

# Import c302 and specific scenario modules
import c302
from c302 import c302_IClamp, c302_Full

# Import Jaxley core and required classes
import jaxley as jx
from jaxley.channels import HH  # Hodgkin-Huxley channel (includes Na, K, Leak):contentReference[oaicite:31]{index=31}
from jaxley.synapses import IonotropicSynapse

# Try importing gap junction mechanism from extension (if available)
try:
    from jaxley_mech.synapses import GapJunction
except ImportError:
    GapJunction = None

def generate_c302_network(preset):
    """
    Generate NeuroML2 network using c302 for the given preset.
    Returns (cell_ids, connections) with normalized, unique cell IDs.
    """
    from types import SimpleNamespace

    def _safe_params(pobj):
        # Ensure we always have .bioparameters and .get_bioparameter(.).value
        if pobj is not None and hasattr(pobj, "bioparameters") and hasattr(pobj, "get_bioparameter"):
            return pobj
        # minimal shim
        def _get(key):
            return SimpleNamespace(value=None)
        return SimpleNamespace(bioparameters={}, get_bioparameter=lambda key: _get(key))

    def _norm_cellref(ref):
        """Normalize NeuroML cell references like 'net/pop[AVAL]' -> 'AVAL'."""
        if ref is None:
            return None
        # Prefer token inside [ ... ]
        if "[" in ref and "]" in ref:
            inside = ref.split("[", 1)[1].split("]", 1)[0]
            if inside:
                return inside
        # Otherwise take last path token
        return ref.split("/")[-1]

    def _iter_chem_connections(proj):
        return getattr(proj, "connections", []) or []

    def _iter_elec_connections(proj):
        for attr in (
            "electrical_connections",
            "electrical_connection_instances",
            "electrical_connection_instance_ws",
            "electricalConnectionInstances",
        ):
            conns = getattr(proj, attr, None)
            if conns:
                return conns
        return []

    def _get_pre_post(conn):
        pre = getattr(conn, "pre_cell_id", None) or getattr(conn, "pre_cell", None)
        post = getattr(conn, "post_cell_id", None) or getattr(conn, "post_cell", None)
        return _norm_cellref(pre), _norm_cellref(post)

    preset = str(preset).upper()

    # ---- 1) Generate NeuroML via c302 per preset ----
    nml_doc = None
    params = None
    cell_ids_hint = []   # any cell list c302 gives us
    try:
        if preset in ("IA", "IB"):
            param_set = preset[1]  # 'A' or 'B'
            # returns: cells_to_stim, all_cells, params, muscles, nml_doc  (common in c302_IClamp)
            out = c302_IClamp.setup(param_set, generate=True, verbose=False)
            # Be flexible about tuple shapes:
            # find the NeuroML doc object in the tuple
            for item in out:
                if hasattr(item, "networks"):
                    nml_doc = item
            # params likely in the tuple: pick first object with 'bioparameters'
            for item in out:
                if hasattr(item, "bioparameters"):
                    params = item
                    break
            # collect any list-like items as hints (cells/muscles)
            for item in out:
                if isinstance(item, (list, tuple)):
                    cell_ids_hint.extend([str(x) for x in item])
        elif preset == "C":
            out = c302_Full.setup("C", generate=True, verbose=False)
            for item in out:
                if hasattr(item, "networks"):
                    nml_doc = item
            for item in out:
                if hasattr(item, "bioparameters"):
                    params = item
                    break
            for item in out:
                if isinstance(item, (list, tuple)):
                    cell_ids_hint.extend([str(x) for x in item])
        elif preset == "FULL":
            out = c302_Full.setup("A", generate=True, verbose=False)
            for item in out:
                if hasattr(item, "networks"):
                    nml_doc = item
            for item in out:
                if hasattr(item, "bioparameters"):
                    params = item
                    break
            for item in out:
                if isinstance(item, (list, tuple)):
                    cell_ids_hint.extend([str(x) for x in item])
        else:
            raise ValueError(f"Unknown preset '{preset}'. Use IA, IB, C, or Full.")
    except Exception as e:
        raise RuntimeError(f"c302 generation failed for preset {preset}: {e}")

    if nml_doc is None or not getattr(nml_doc, "networks", None):
        raise RuntimeError("No NeuroML document/networks returned by c302.")

    params = _safe_params(params)
    net = nml_doc.networks[0]

    pop_ids = [str(getattr(p, "id", "")) for p in getattr(net, "populations", []) if getattr(p, "id", None)]
    pop_id_set = set(pop_ids)

    # ---- 2) Parse connections (chemical + electrical) ----
    connections = []

    # From net.projections (mostly chemical, but be defensive)
    for proj in getattr(net, "projections", []):
        # Chemical
        conns = _iter_chem_connections(proj)
        if conns:
            synapse_id = getattr(proj, "synapse", "") or ""
            sid_lower = str(synapse_id).lower()
            for conn in conns:
                pre, post = _get_pre_post(conn)
                if pre is None or post is None:
                    continue
                syn_type = "exc"
                if "inh" in sid_lower:
                    syn_type = "inh"
                elif "exc" in sid_lower:
                    syn_type = "exc"
                # gbase from params
                key = "neuron_to_neuron_chem_exc_syn_gbase" if syn_type == "exc" else "neuron_to_neuron_chem_inh_syn_gbase"
                g_base = None
                if key in params.bioparameters:
                    g_base = params.get_bioparameter(key).value
                # fallback
                try:
                    g_val = float(str(g_base).split()[0]) if g_base else 0.1
                except Exception:
                    g_val = 0.1
                connections.append({"pre": pre, "post": post, "type": syn_type, "g": g_val * 1e-9})
            continue
        # Electrical sneaking under projections
        elec = _iter_elec_connections(proj)
        if elec:
            for conn in elec:
                pre, post = _get_pre_post(conn)
                if pre is None or post is None:
                    continue
                g_gap = None
                if "neuron_to_neuron_elec_syn_gbase" in params.bioparameters:
                    g_gap = params.get_bioparameter("neuron_to_neuron_elec_syn_gbase").value
                try:
                    g_val = float(str(g_gap).split()[0]) if g_gap else 0.0005
                except Exception:
                    g_val = 0.0005
                connections.append({"pre": pre, "post": post, "type": "gap", "g": g_val * 1e-9})

    # From net.electrical_projections (canonical place)
    for proj in getattr(net, "electrical_projections", []):
        for conn in _iter_elec_connections(proj):
            pre, post = _get_pre_post(conn)
            if pre is None or post is None:
                continue
            g_gap = None
            if "neuron_to_neuron_elec_syn_gbase" in params.bioparameters:
                g_gap = params.get_bioparameter("neuron_to_neuron_elec_syn_gbase").value
            try:
                g_val = float(str(g_gap).split()[0]) if g_gap else 0.0005
            except Exception:
                g_val = 0.0005
            connections.append({"pre": pre, "post": post, "type": "gap", "g": g_val * 1e-9})

    # ---- 3) Build cell_ids robustly ----
    # Prefer explicit lists from c302 (cell_names, muscles, etc.)
    norm_hints = [_norm_cellref(x) for x in cell_ids_hint if x is not None]
    norm_hints = [x for x in norm_hints if x]  # drop Nones/empties

    # Add any ids visible in connections
    ids_from_conns = []
    for c in connections:
        if c["pre"]:
            ids_from_conns.append(c["pre"])
        if c["post"]:
            ids_from_conns.append(c["post"])

    # Try to glean from populations (if present)
    ids_from_pops = []
    for pop in getattr(net, "populations", []):
        pop_id = getattr(pop, "id", None)
        # Instances may carry instance ids; normalize them
        instances = getattr(pop, "instances", None) or []
        if instances:
            for inst in instances:
                inst_id = getattr(inst, "id", None)
                if inst_id is not None:
                    ids_from_pops.append(str(inst_id))
        elif pop_id:
            # fallback: at least include population id (better than nothing)
            ids_from_pops.append(str(pop_id))

    # Merge and de-dup while preserving order
    seen = set()
    merged = []
    for origin in (norm_hints, ids_from_conns, ids_from_pops):
        for cid in origin:
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(cid)

    cell_ids = merged
    if not cell_ids:
        raise RuntimeError("Could not determine any cell IDs from c302 output or NeuroML.")

    # Optional sanity print
    # print(f"[generate_c302_network] cells={len(cell_ids)}, conns={len(connections)}")

    return cell_ids, connections

def build_jaxley_network(cell_ids, connections):
    """
    Build a Jaxley network from cell list and connection list.
    Returns the jx.Network object and a dictionary of cell objects.
    """
    num_cells = len(cell_ids)
    # Create one compartment cell for each cell_id
    cell_objects = {}
    jx_cells = []
    for cid in cell_ids:
        comp = jx.Compartment()
        branch = jx.Branch(comp, ncomp=1)   # single compartment branch
        cell = jx.Cell(branch, parents=[-1])
        # Insert Hodgkin-Huxley channel mechanism into the cell (gives Na, K, Leak)
        cell.insert(HH())
        cell_objects[cid] = cell
        jx_cells.append(cell)
        print("Cell " + cid + " created.")
    # Create the network with all cells
    net = jx.Network(jx_cells)
    # Connect synapses according to the connections list
    connectionsLength = len(connections)
    print("Connections length: " + str(connectionsLength))
    i = 0 
    for conn in connections:
        i += 1
        pre_id = conn["pre"]
        post_id = conn["post"]
        syn_type = conn["type"]
        g_val = conn["g"]

        print("Connection "+str(i) + "/" + str(connectionsLength) + ": "+pre_id + "-->" + post_id)

        try:
            pre_comp = net.cell(cell_ids.index(pre_id)).branch(0).loc(0.0)
            post_comp = net.cell(cell_ids.index(post_id)).branch(0).loc(0.0)
        except Exception as e:
            # If cell not found (should not happen if cell_ids list is complete)
            continue
        if syn_type in ("exc", "inh"):
            # Chemical synapse: use IonotropicSynapse
            syn = IonotropicSynapse()
            jx.connect(pre_comp, post_comp, syn)
            # Set synaptic conductance (gS) and reversal (e_syn)
            # Find the index of the newly added synapse (last edge in net.IonotropicSynapse)
            edge_idx = net.edges.index[-1]  # index of the last added edge
            # Set parameters for that synapse edge
            rev_potential = 0.0 if syn_type == "exc" else -70.0  # mV
            net.IonotropicSynapse.edge(edge_idx).set("IonotropicSynapse_gS", g_val)      # in S
            net.IonotropicSynapse.edge(edge_idx).set("IonotropicSynapse_e_syn", rev_potential)
        elif syn_type == "gap" and GapJunction is not None:
            # Electrical synapse: use GapJunction if available
            syn = GapJunction()  # electrical coupling
            jx.connect(pre_comp, post_comp, syn)
            # Set coupling conductance (gGap or similar parameter name)
            edge_idx = net.edges.index[-1]
            try:
                net.GapJunction.edge(edge_idx).set("GapJunction_gGap", g_val)
            except Exception:
                # If parameter name differs or direct setting fails, set all edges as fallback
                net.set("GapJunction_gGap", g_val)
        else:
            # Gap junction but no GapJunction class available; skip or approximate
            pass
    return net, cell_objects

def run_simulation(net, cell_ids, record_subset=None, duration=1000.0, dt=0.05):
    """
    Run the Jaxley simulation for the given network.
    If record_subset is provided, only records those cells; otherwise records all cells.
    Returns time array and a dict of recorded voltage traces.
    """
    # Configure recording
    recorded_ids = record_subset if record_subset else cell_ids[:]
    record_indices = [cell_ids.index(cid) for cid in recorded_ids if cid in cell_ids]
    for i in record_indices:
        net.cell(i).branch(0).loc(0.0).record("v")  # record membrane voltage
    # Run the simulation
    print(f"Running simulation for {duration} ms with dt={dt} ms...")
    V = jx.integrate(net, delta_t=dt, t_max=duration)
    # V will be an array of shape (num_recorded, num_timepoints).
    # Extract time vector and traces:
    time = np.arange(0, duration+dt, dt)
    traces = {}
    for idx, cid in zip(record_indices, recorded_ids):
        # jx.integrate returns recordings in the order they were added
        # Here we assume they were added in same order as record_indices
        traces[cid] = V[len(traces)]  # each row in V corresponds to one recorded loc
    return time, traces

def main():
    parser = argparse.ArgumentParser(description="Run c302 connectome in Jaxley")
    parser.add_argument("--preset", type=str, required=True, 
                        help="c302 preset level: IA, IB, C, or Full")
    parser.add_argument("--duration", type=float, default=1000.0, 
                        help="Simulation duration in ms (default 1000)")
    parser.add_argument("--dt", type=float, default=0.05, 
                        help="Timestep in ms (default 0.05)")
    parser.add_argument("--plot_cells", type=int, default=5, 
                        help="Number of cells to plot (default 5)")
    args = parser.parse_args()
    preset = args.preset
    duration = args.duration
    dt = args.dt
    # 1. Generate network via c302
    cell_ids, connections = generate_c302_network(preset)
    # 2. Build Jaxley network
    net, cell_objs = build_jaxley_network(cell_ids, connections)
    # 3. Run simulation
    # Select subset of cells to record/plot (e.g. first N cells)
    subset = cell_ids[:args.plot_cells]
    time, traces = run_simulation(net, cell_ids, record_subset=subset, duration=duration, dt=dt)
    # 4. Output summary and plot results
    num_neurons = sum(1 for cid in cell_ids if not cid.startswith("M"))  # crude check: muscle cells often start with 'M'
    num_muscles = len(cell_ids) - num_neurons
    num_gap = sum(1 for conn in connections if conn["type"] == "gap")
    num_chem = len(connections) - num_gap
    num_exc = sum(1 for conn in connections if conn["type"] == "exc")
    num_inh = sum(1 for conn in connections if conn["type"] == "inh")
    print(f"Network contains {num_neurons} neurons and {num_muscles} muscles.")
    print(f"Total synaptic connections: {len(connections)}  (chemical: {num_chem} [{num_exc} excitatory, {num_inh} inhibitory], electrical (gap junction): {num_gap})")
    print(f"Recorded {len(traces)} cells: {list(traces.keys())}")
    # Plot the recorded traces
    plt.figure(figsize=(8, 6))
    for cid, V_trace in traces.items():
        plt.plot(time, V_trace, label=f"{cid}")
    plt.title(f"Voltage traces ({preset} preset, {duration} ms simulation)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.tight_layout()
    # Save plot to file
    plot_file = f"voltage_traces_{preset}.png"
    plt.savefig(plot_file)
    print(f"Voltage trace plot saved to {plot_file}")

if __name__ == "__main__":
    main()
