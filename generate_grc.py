"""
=============================================================================
SADM-SEC  |  generate_grc.py
=============================================================================
GRC Flowgraph Generator
Generates the sadm_flowgraph.grc file which can be opened directly in
GNU Radio Companion.

The flowgraph topology:
  ┌─────────────┐   ┌────────────┐   ┌────────────────────┐
  │ Audio Source│──▶│  Multiply  │──▶│                    │   8× ZMQ PUB
  │  (Mic 48kHz)│   │ (DSB-SC   )│   │  SADM Beamformer   │──▶ Sinks
  └─────────────┘   └────────────┘   │  (Embedded Python  │   ports 5555–5562
  ┌─────────────┐        ▲           │   Block)           │
  │ Signal Src  │────────┘           │                    │
  │(carrier tone│                    └────────────────────┘
  │  10 kHz)    │   ┌──────────────▶│ in1 (noise)        │
  └─────────────┘   │               └────────────────────┘
  ┌─────────────┐   │
  │ Noise Source│───┘
  └─────────────┘

Run this script to create the .grc file:
    python generate_grc.py

Then open sadm_flowgraph.grc in GNU Radio Companion.
=============================================================================
"""

# Embedded source code for the SADM beamformer block
SADM_BEAMFORMER_SOURCE = '''import numpy as np
import os
import sys
from gnuradio import gr

# Attempt to import our math engine from the same directory.
# In GRC, add the sadm_sec/ folder to the GRC Python path via
# Tools → Options → General → "Additional Python Paths".
# Also try adding the script directory to path for embedded blocks.
try:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _current_dir = os.path.abspath(".")
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from spatial_logic import (
        N_ANTENNAS, beamforming_weights,
        noise_projection_matrix, SADMTracker
    )
    _SPATIAL_OK = True
except ImportError as e:
    _SPATIAL_OK = False
    print(f"[SADM Block] Warning: Could not import spatial_logic: {e}")


class sadm_beamformer(gr.sync_block):
    \"\"\"
    SADM Physical-Layer Security Block

    Takes 1 message stream + 1 noise stream and outputs
    N_ANTENNAS = 8 spatially shaped streams.
    \"\"\"

    def __init__(self,
                 bob_angle: float = 30.0,
                 snr_sig_db: float = 20.0,
                 snr_an_db: float  = 10.0,
                 n_antennas: int   = 8):

        gr.sync_block.__init__(
            self,
            name="SADM Beamformer",
            in_sig  = [np.complex64, np.complex64],          # msg, noise
            out_sig = [np.complex64] * n_antennas             # 8 antennas
        )

        self.N          = n_antennas
        self.snr_sig    = 10 ** (snr_sig_db / 10)
        self.snr_an     = 10 ** (snr_an_db  / 10)

        if _SPATIAL_OK:
            self.tracker = SADMTracker(bob_initial_angle=bob_angle,
                                       n_antennas=n_antennas,
                                       alpha=0.3)
        else:
            # Fallback: pre-compute static weights
            theta = np.deg2rad(bob_angle)
            k     = np.arange(n_antennas)
            a     = np.exp(1j * np.pi * np.sin(theta) * k)
            self.w_static    = a / np.linalg.norm(a)
            self.P_AN_static = np.eye(n_antennas, dtype=complex)
            print("[SADM Block] spatial_logic.py not found - "
                  "using static beamforming fallback.")

    # ── Called by GRC framework to update the steering angle ─────────────────
    def set_bob_angle(self, angle_deg: float) -> None:
        if _SPATIAL_OK:
            self.tracker.angle = angle_deg
            self.tracker._update_weights()
        else:
            theta  = np.deg2rad(angle_deg)
            k      = np.arange(self.N)
            a      = np.exp(1j * np.pi * np.sin(theta) * k)
            self.w_static = a / np.linalg.norm(a)

    # ── Called by GRC framework to update signal power ──────────────────────────
    def set_snr_sig_db(self, snr_db: float) -> None:
        \"\"\"Update the signal power ratio (in dB).\"\"\"
        self.snr_sig = 10 ** (snr_db / 10)

    # ── Called by GRC framework to update AN power ──────────────────────────────
    def set_snr_an_db(self, snr_db: float) -> None:
        \"\"\"Update the artificial noise power ratio (in dB).\"\"\"
        self.snr_an = 10 ** (snr_db / 10)

    # ── Main processing function ──────────────────────────────────────────────
    def work(self, input_items, output_items):
        msg   = input_items[0]          # (L,) complex64
        noise = input_items[1]          # (L,) complex64
        L     = len(msg)

        if _SPATIAL_OK:
            w    = self.tracker.w                    # (N,)
            P_AN = self.tracker.P_AN                 # (N,N)
        else:
            w    = self.w_static
            P_AN = self.P_AN_static

        # Message stream → beamformed signal
        msg_part = np.outer(w, msg) * np.sqrt(self.snr_sig)     # (N, L)

        # Noise stream → projected artificial noise
        noise_mat = np.outer(np.ones(self.N), noise)             # (N, L)
        an_part   = (P_AN @ noise_mat) * np.sqrt(self.snr_an)   # (N, L)

        transmit = (msg_part + an_part).astype(np.complex64)     # (N, L)

        # Write to output ports
        for i in range(self.N):
            output_items[i][:L] = transmit[i]

        return L
'''

GRC_TEMPLATE = '''<?xml version='1.0' encoding='utf-8'?>
<flow_graph>
  <metadata>
    <format>1</format>
    <generator>SADM-SEC generate_grc.py</generator>
  </metadata>

  <block>
    <name>options</name>
    <key>options</key>
    <param><key>author</key><value>SADM-SEC Project</value></param>
    <param><key>catch_exceptions</key><value>True</value></param>
    <param><key>category</key><value>[GRC Hier Blocks]</value></param>
    <param><key>generate_options</key><value>qt_gui</value></param>
    <param><key>hier_block_src_path</key><value>.:</value></param>
    <param><key>id</key><value>sadm_flowgraph</value></param>
    <param><key>max_nouts</key><value>0</value></param>
    <param><key>output_language</key><value>python</value></param>
    <param><key>placement</key><value>(0,0)</value></param>
    <param><key>qt_qss_theme</key><value></value></param>
    <param><key>realtime_scheduling</key><value></value></param>
    <param><key>run</key><value>True</value></param>
    <param><key>run_command</key><value>{python} -u {filename}</value></param>
    <param><key>run_options</key><value>prompt</value></param>
    <param><key>sizing_mode</key><value>fixed</value></param>
    <param><key>thread_safe_setters</key><value></value></param>
    <param><key>title</key><value>SADM-SEC Physical Layer Security</value></param>
    <param><key>window_size</key><value>(1000,1000)</value></param>
  </block>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!--  VARIABLES                                                          -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <block>
    <name>samp_rate</name>
    <key>variable</key>
    <param><key>id</key><value>samp_rate</value></param>
    <param><key>value</key><value>48000</value></param>
  </block>
  <block>
    <name>carrier_freq</name>
    <key>variable</key>
    <param><key>id</key><value>carrier_freq</value></param>
    <param><key>value</key><value>10000</value></param>
  </block>
  <block>
    <name>bob_angle</name>
    <key>variable_qtgui_range</key>
    <param><key>id</key><value>bob_angle</value></param>
    <param><key>value</key><value>30</value></param>
    <param><key>start</key><value>-90</value></param>
    <param><key>stop</key><value>90</value></param>
    <param><key>step</key><value>1</value></param>
    <param><key>label</key><value>Bob Angle (degrees)</value></param>
    <param><key>widget</key><value>slider</value></param>
  </block>
  <block>
    <name>snr_signal_db</name>
    <key>variable_qtgui_range</key>
    <param><key>id</key><value>snr_signal_db</value></param>
    <param><key>value</key><value>20</value></param>
    <param><key>start</key><value>0</value></param>
    <param><key>stop</key><value>40</value></param>
    <param><key>step</key><value>1</value></param>
    <param><key>label</key><value>Signal Power (dB)</value></param>
    <param><key>widget</key><value>slider</value></param>
  </block>
  <block>
    <name>snr_an_db</name>
    <key>variable_qtgui_range</key>
    <param><key>id</key><value>snr_an_db</value></param>
    <param><key>value</key><value>10</value></param>
    <param><key>start</key><value>0</value></param>
    <param><key>stop</key><value>30</value></param>
    <param><key>step</key><value>1</value></param>
    <param><key>label</key><value>Artificial Noise Power (dB)</value></param>
    <param><key>widget</key><value>slider</value></param>
  </block>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!--  AUDIO SOURCE  (Microphone)                                         -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <block>
    <name>audio_source_0</name>
    <key>audio_source</key>
    <param><key>affinity</key><value></value></param>
    <param><key>alias</key><value></value></param>
    <param><key>comment</key><value>Microphone input (DSB-SC source)</value></param>
    <param><key>device_name</key><value></value></param>
    <param><key>id</key><value>audio_source_0</value></param>
    <param><key>num_inputs</key><value>1</value></param>
    <param><key>ok_to_block</key><value>True</value></param>
    <param><key>samp_rate</key><value>samp_rate</value></param>
  </block>

  <!-- Float-to-Complex for audio -->
  <block>
    <name>blocks_float_to_complex_0</name>
    <key>blocks_float_to_complex</key>
    <param><key>id</key><value>blocks_float_to_complex_0</value></param>
    <param><key>vlen</key><value>1</value></param>
  </block>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!--  CARRIER SIGNAL SOURCE  (DSB-SC modulation)                         -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <block>
    <name>analog_sig_source_carrier</name>
    <key>analog_sig_source_x</key>
    <param><key>id</key><value>analog_sig_source_carrier</value></param>
    <param><key>type</key><value>complex</value></param>
    <param><key>samp_rate</key><value>samp_rate</value></param>
    <param><key>waveform</key><value>analog.WAVEFORM_COSINE</value></param>
    <param><key>freq</key><value>carrier_freq</value></param>
    <param><key>amp</key><value>1</value></param>
    <param><key>offset</key><value>0</value></param>
  </block>

  <!-- Multiply: DSB-SC modulation = audio × carrier -->
  <block>
    <name>blocks_multiply_dsb</name>
    <key>blocks_multiply_xx</key>
    <param><key>id</key><value>blocks_multiply_dsb</value></param>
    <param><key>type</key><value>complex</value></param>
    <param><key>num_inputs</key><value>2</value></param>
    <param><key>vlen</key><value>1</value></param>
  </block>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!--  NOISE SOURCE  (for Artificial Noise injection)                     -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <block>
    <name>analog_noise_source_an</name>
    <key>analog_noise_source_x</key>
    <param><key>id</key><value>analog_noise_source_an</value></param>
    <param><key>type</key><value>complex</value></param>
    <param><key>noise_type</key><value>analog.GR_GAUSSIAN</value></param>
    <param><key>amplitude</key><value>1</value></param>
    <param><key>seed</key><value>0</value></param>
  </block>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!--  SADM BEAMFORMER  (Embedded Python Block)                           -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
'''

# Dynamic part with source code embedded
SADM_BLOCK_MIDDLE = '''  <block>
    <name>sadm_beamformer_0</name>
    <key>epy_block</key>
    <param><key>id</key><value>sadm_beamformer_0</value></param>
    <param><key>_source_code</key><value>{source_code}</value></param>
    <param><key>bob_angle</key><value>bob_angle</value></param>
    <param><key>snr_sig_db</key><value>snr_signal_db</value></param>
    <param><key>snr_an_db</key><value>snr_an_db</value></param>
    <param><key>n_antennas</key><value>8</value></param>
  </block>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!--  ZMQ PUB SINKS  (one per antenna, ports 5555–5562)                -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
'''

# Dynamically add 8 ZMQ PUB sink blocks
ZMQ_BLOCK_TEMPLATE = '''  <block>
    <name>zeromq_pub_sink_{i}</name>
    <key>zeromq_pub_sink</key>
    <param><key>id</key><value>zeromq_pub_sink_{i}</value></param>
    <param><key>type</key><value>complex</value></param>
    <param><key>address</key><value>tcp://127.0.0.1:{port}</value></param>
    <param><key>vlen</key><value>1</value></param>
    <param><key>pass_tags</key><value>False</value></param>
    <param><key>timeout</key><value>100</value></param>
    <param><key>hwm</key><value>-1</value></param>
  </block>
'''

GRC_CONNECTIONS = '''  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!--  CONNECTIONS                                                        -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!-- audio → float_to_complex -->
  <connection>
    <source_block_id>audio_source_0</source_block_id>
    <source_key>0</source_key>
    <sink_block_id>blocks_float_to_complex_0</sink_block_id>
    <sink_key>0</sink_key>
  </connection>
  <!-- float_to_complex → multiply (DSB-SC in0) -->
  <connection>
    <source_block_id>blocks_float_to_complex_0</source_block_id>
    <source_key>0</source_key>
    <sink_block_id>blocks_multiply_dsb</sink_block_id>
    <sink_key>0</sink_key>
  </connection>
  <!-- carrier → multiply (DSB-SC in1) -->
  <connection>
    <source_block_id>analog_sig_source_carrier</source_block_id>
    <source_key>0</source_key>
    <sink_block_id>blocks_multiply_dsb</sink_block_id>
    <sink_key>1</sink_key>
  </connection>
  <!-- DSB-SC signal → SADM block in0 (message) -->
  <connection>
    <source_block_id>blocks_multiply_dsb</source_block_id>
    <source_key>0</source_key>
    <sink_block_id>sadm_beamformer_0</sink_block_id>
    <sink_key>0</sink_key>
  </connection>
  <!-- Noise → SADM block in1 (artificial noise) -->
  <connection>
    <source_block_id>analog_noise_source_an</source_block_id>
    <source_key>0</source_key>
    <sink_block_id>sadm_beamformer_0</sink_block_id>
    <sink_key>1</sink_key>
  </connection>
'''

ZMQ_CONNECTION_TEMPLATE = '''  <connection>
    <source_block_id>sadm_beamformer_0</source_block_id>
    <source_key>{i}</source_key>
    <sink_block_id>zeromq_pub_sink_{i}</sink_block_id>
    <sink_key>0</sink_key>
  </connection>
'''

GRC_FOOTER = "</flow_graph>\n"


def xml_escape(s: str) -> str:
    """Escape special XML characters in source code."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))


def generate_grc(output_path: str = "sadm_flowgraph.grc") -> None:
    """Write the complete GRC XML file."""
    parts = [GRC_TEMPLATE]

    # Add SADM block with embedded source code (properly XML-escaped)
    parts.append(SADM_BLOCK_MIDDLE.format(source_code=xml_escape(SADM_BEAMFORMER_SOURCE)))

    # 8 ZMQ PUB sink block definitions
    for i in range(8):
        parts.append(ZMQ_BLOCK_TEMPLATE.format(i=i, port=5555 + i))

    parts.append(GRC_CONNECTIONS)

    # 8 ZMQ connections
    for i in range(8):
        parts.append(ZMQ_CONNECTION_TEMPLATE.format(i=i))

    parts.append(GRC_FOOTER)

    grc_xml = "".join(parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(grc_xml)

    print(f"[GRC Generator] Flowgraph written to: {output_path}")
    print("  Open with:  gnuradio-companion sadm_flowgraph.grc")
    print("  Requires GNU Radio 3.10+ and the zeromq OOT module.")


if __name__ == "__main__":
    generate_grc()
