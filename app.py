from __future__ import annotations

import streamlit as st

from generate import generate


st.set_page_config(
    page_title="MiniGPT – Transformer Language Model",
    page_icon="🧠",
    layout="centered",
)


def main() -> None:
    st.title("MiniGPT – Transformer Language Model")

    st.markdown(
        "Enter a starting prompt below and MiniGPT will generate a continuation "
        "using a decoder-only Transformer trained on your dataset."
    )

    with st.form("generation_form"):
        prompt = st.text_area(
            "Starting prompt",
            value="",
            height=150,
            placeholder="Type the beginning of your text here...",
        )

        max_tokens = st.slider(
            "Generation length (tokens)",
            min_value=10,
            max_value=500,
            value=80,
            step=10,
        )

        submitted = st.form_submit_button("Generate Text")

    if submitted:
        if not prompt.strip():
            st.warning("Please provide a non-empty starting prompt.")
        else:
            with st.spinner("Generating..."):
                try:
                    output = generate(prompt=prompt, max_new_tokens=max_tokens)
                except Exception as exc:  # streamlit-friendly error display
                    st.error(f"Generation failed: {exc}")
                else:
                    st.markdown("#### Generated output")
                    st.markdown(
                        """
                        <div style="
                            padding: 1rem 1.25rem;
                            border-radius: 0.5rem;
                            background-color: #111827;
                            color: #e5e7eb;
                            border: 1px solid #374151;
                            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                            white-space: pre-wrap;
                        ">
                        {text}
                        </div>
                        """.format(text=output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")),
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #6b7280; font-size: 0.9rem;">'
        "Built from scratch using PyTorch Transformer"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

