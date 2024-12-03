import streamlit as st
from math import gcd
from random import randint
import numpy as np
from qiskit import *
from qiskit_aer import AerSimulator
import RSA_module

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Streamlit UI with Sidebar Navigation
st.title("Quantum Cyrptography")

# Sidebar for Navigation
option = st.sidebar.radio("Select Algorithm", ("Quantum Key Distribution", "Asymmetric Encryption"))

if option == "Quantum Key Distribution":

    # Initialize session state variables
    if 'alice_bits' not in st.session_state:
        st.session_state.alice_bits = None
        st.session_state.alice_bases = None
        st.session_state.bob_bases = None
        st.session_state.bob_results = None
        st.session_state.eve_intercept = False

    # Function to encode the message
    def encode_message(bits, bases, n):
        print("Executing encode_message function...")
        message = []
        for i in range(n):
            qc = QuantumCircuit(1, 1)
            if bases[i] == 0:  # Prepare qubit in Z-basis
                if bits[i] == 0:
                    pass
                else:
                    qc.x(0)
            else:  # Prepare qubit in X-basis
                if bits[i] == 0:
                    qc.h(0)
                else:
                    qc.x(0)
                    qc.h(0)
            qc.barrier()
            message.append(qc)
        print("Message encoding complete.")
        return message

    # Function to measure the message
    def measure_message(message, bases, n):
        print("Executing measure_message function...")
        backend = Aer.get_backend('aer_simulator')
        measurements = []
        for q in range(n):
            if bases[q] == 0:  # Measuring in Z-basis
                message[q].measure(0, 0)
            if bases[q] == 1:  # Measuring in X-basis
                message[q].h(0)
                message[q].measure(0, 0)
            result = backend.run(message[q], shots=1, memory=True).result()
            measured_bit = int(result.get_memory()[0])
            measurements.append(measured_bit)
        print(f"Measurement results: {measurements}")
        return measurements

    # Function to perform sifting (removing non-matching bases)
    def remove_unmatching_sift_keys(a_bases, b_bases, bits, n):
        print("Executing remove_unmatching_sift_keys function...")
        good_bits = []
        for q in range(n):
            if a_bases[q] == b_bases[q]:
                good_bits.append(bits[q])
        print(f"Sifted bits: {good_bits}")
        return good_bits

    # GUI layout
    st.title("BB84 Quantum Key Distribution Protocol")

    # Alice's key generation
    n = 10 # Number of bits
    if st.button("Generate Alice's Key"):
        st.session_state.alice_bits = np.random.randint(0, 2, n)
        st.session_state.alice_bases = np.random.randint(0, 2, n)
        st.write(f"Alice's Bits: {st.session_state.alice_bits}")
        st.write(f"Alice's Bases: {st.session_state.alice_bases}")

    # Eve's interception toggle
    st.session_state.eve_intercept = st.radio("Eve Intercept?", [False, True])

    # Bob's measurement
    if st.button("Bob Measures"):
        print("Bob Measures button clicked.")
        st.session_state.bob_bases = np.random.randint(0, 2, n)
        message = encode_message(st.session_state.alice_bits, st.session_state.alice_bases, n)
        if st.session_state.eve_intercept:
            print("Eve is intercepting the message.")
            eve_bases = np.random.randint(0, 2, n)
            intercepted_message = measure_message(message, eve_bases, n)
            st.write(f"Eve's Bases: {eve_bases}")
            st.write(f"Eve's Results: {intercepted_message}")
            message = encode_message(intercepted_message, st.session_state.alice_bases, n)
        st.session_state.bob_results = measure_message(message, st.session_state.bob_bases, n)
        st.write(f"Bob's Bases: {st.session_state.bob_bases}")
        st.write(f"Bob's Results: {st.session_state.bob_results}")

    # Sifting process
    if st.button("Perform Sifting"):
        # Match Alice's and Bob's bases
        matching_bases = st.session_state.alice_bases == st.session_state.bob_bases
        matching_indices = np.where(matching_bases)[0]  # Get indices where the bases match
        
        # Convert bob_results to a numpy array if it's not already
        bob_results_array = np.array(st.session_state.bob_results)
        
        # Extract keys using the matching indices
        alice_key = st.session_state.alice_bits[matching_indices]
        bob_key = bob_results_array[matching_indices]
        
        st.write(f"Sifted Alice's Key: {alice_key}")
        st.write(f"Sifted Bob's Key: {bob_key}")

        # Error detection
        errors = np.sum(alice_key != bob_key)
        error_rate = errors / len(alice_key) * 100
        st.write(f"Error Rate: {error_rate:.2f}%")
        if st.session_state.eve_intercept and error_rate > 0:
            st.write("Eve Detected!")
        else:
            st.write("No Eavesdropping Detected!")

elif option == "Asymmetric Encryption":
    # Set up the quantum simulator for Shor's algorithm
    qasm_sim = AerSimulator()

    # Function for period finding (Shor's algorithm)
    def period(a, N):
        available_qubits = 16
        r = -1

        if N >= 2**available_qubits:
            print(str(N) + ' is too big for IBMQX')

        qr = QuantumRegister(available_qubits)
        cr = ClassicalRegister(available_qubits)
        qc = QuantumCircuit(qr, cr)
        x0 = randint(1, N-1)
        x_binary = np.zeros(available_qubits, dtype=bool)

        for i in range(1, available_qubits + 1):
            bit_state = (N % (2**i) != 0)
            if bit_state:
                N -= 2**(i - 1)
            x_binary[available_qubits - i] = bit_state

        for i in range(0, available_qubits):
            if x_binary[available_qubits - i - 1]:
                qc.x(qr[i])
        x = x0

        while np.logical_or(x != x0, r <= 0):
            r += 1
            qc.measure(qr, cr)
            for i in range(0, 3):
                qc.x(qr[i])
            qc.cx(qr[2], qr[1])
            qc.cx(qr[1], qr[2])
            qc.cx(qr[2], qr[1])
            qc.cx(qr[1], qr[0])
            qc.cx(qr[0], qr[1])
            qc.cx(qr[1], qr[0])
            qc.cx(qr[3], qr[0])
            qc.cx(qr[0], qr[1])
            qc.cx(qr[1], qr[0])

            result = qasm_sim.run(qc).result()
            counts = result.get_counts()

            results = [[], []]
            for key, value in counts.items():
                results[0].append(key)
                results[1].append(int(value))
            s = results[0][np.argmax(np.array(results[1]))]
        return r

    # Shor's algorithm implementation
    def shors_breaker(N):
        N = int(N)
        while True:
            a = randint(0, N - 1)
            g = gcd(a, N)
            if g != 1 or N == 1:
                return g, N // g
            else:
                r = period(a, N)
                if r % 2 != 0:
                    continue
                elif pow(a, r // 2, N) == -1:
                    continue
                else:
                    p = gcd(pow(a, r // 2) + 1, N)
                    q = gcd(pow(a, r // 2) - 1, N)
                    if p == N or q == N:
                        continue
                    return p, q

    # Function to find modular inverse
    def modular_inverse(a, m):
        a = a % m
        for x in range(1, m):
            if ((a * x) % m == 1):
                return x
        return 1
    st.subheader("Asymmetric Encryption (RSA)")

    # RSA Encryption setup
    bit_length = st.number_input("Enter bit length for RSA key generation", min_value=1, step=1)
    msg = st.text_area("Write your message to encrypt")
    public, private= 0,0
    # Encrypt and Decrypt with Shor's algorithm button
    if st.button("Encrypt and Decrypt with Shor's Algorithm"):
        if bit_length > 0 and msg:
            # Generate RSA keys
            public, private = RSA_module.generate_keypair(2**bit_length)
            
            # Display public key
            st.write("Public Key: ", public)

            # Encrypt the message
            encrypted_msg, encryption_obj = RSA_module.encrypt(msg, public)
            st.write("Encrypted message: ", encrypted_msg)

            # Use Shor's Algorithm for decryption
            N_shor = public[1]
            p, q = shors_breaker(N_shor)
            phi = (p - 1) * (q - 1)
            d_shor = modular_inverse(public[0], phi)

            # Decrypt using Shor's Algorithm
            decrypted_msg_shor = RSA_module.decrypt(encryption_obj, (d_shor, N_shor))
            st.write("Message Cracked using Shor's Algorithm: ", decrypted_msg_shor)

        else:
            st.error("Please provide valid input for bit length and message.")
