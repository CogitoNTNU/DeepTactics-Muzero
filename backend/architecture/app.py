import streamlit as st
import torch
from network import RepresentationNetwork, PredictionNetwork, DynamicsNetwork


#does the same thing as network.py but in a streamlit app 

# Instantiate networks
rep_net = RepresentationNetwork()
pred_net = PredictionNetwork()
dyn_net = DynamicsNetwork()

# Dummy input
obs = torch.randn(1, 3, 84, 84)
hidden_state = rep_net(obs)

st.title("🧠 MuZero Network Debugger")

if st.button("Test Networks"):
    # Representation
    st.write("🔍 Hidden State:", hidden_state.shape)

    # Prediction
    policy, value = pred_net(hidden_state)
    st.write("🎯 Policy:", policy.tolist())
    st.write("📈 Value:", value.item())

    # Dynamics
    action = torch.tensor([1])
    next_state, reward = dyn_net(hidden_state, action)
    st.write("🔄 Next State Shape:", next_state.shape)
    st.write("🏆 Reward:", reward.item())
