import streamlit as st
import subprocess

def main():
    st.title("Stock Market Prediction")

    # Allow user to input file path
    file_path = st.text_input("Enter the file path:")

    if st.button("Run Prediction"):
        if file_path:
            try:
                # Remove double quotes from file path
                file_path = file_path.strip('"')

                # Run the backend script with the provided file path
                result = subprocess.run(["python", "backend.py", file_path], capture_output=True, text=True, check=True)
                
                # Split the output into lines
                output_lines = result.stdout.strip().split('\n')
                
                # Display evaluation metrics
                st.subheader("Evaluation Metrics:")
                for line in output_lines:
                    st.write(line)
                
                # Display the plot image
                st.subheader("Actual vs Predicted Closing Prices:")
                plot_path = output_lines[-1]  # Last line contains path to plot image
                st.image(plot_path)
                
            except subprocess.CalledProcessError as e:
                st.error(f"Error running prediction: {e}")
        else:
            st.error("Please enter a valid file path.")

if __name__ == "__main__":
    main()