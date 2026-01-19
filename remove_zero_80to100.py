import pandas as pd
import matplotlib.pyplot as plt

def process_gpu_data(input_file, output_file, plot_file):
    # 1. Read the CSV file, skipping comment lines starting with '#'
    try:
        df = pd.read_csv(input_file, comment='#', skip_blank_lines=True)
        df.columns = df.columns.str.strip()  # clean column names
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # --- SETTINGS ---
    gpu_col = 'GPU (%)'
    time_col = 'Time (s)'
    # ----------------

    # 2. Check if the GPU column exists
    if gpu_col not in df.columns:
        print(f"Error: Column '{gpu_col}' not found.")
        print(f"Found columns: {list(df.columns)}")
        return

    # 3. Filter out entries where GPU (%) is 0
    df[gpu_col] = pd.to_numeric(df[gpu_col], errors='coerce')
    filtered_df = df[df[gpu_col] != 0].copy()

    # 4. Save the filtered data to a new CSV
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

    # 5. Generate a plot
    if not filtered_df.empty:
        plt.figure(figsize=(16, 6))  # wider plot

        x_col = time_col if time_col in filtered_df.columns else None
        if x_col:
            filtered_df = filtered_df.sort_values(x_col)

            plt.plot(
                filtered_df[x_col], filtered_df[gpu_col],
                linestyle='-', color='b',
                linewidth=1.8, alpha=0.9
            )
            plt.xlabel(x_col)
            plt.xlim(filtered_df[x_col].min() - 1, filtered_df[x_col].max() + 1)
            plt.xticks(rotation=45)
            plt.locator_params(axis='x', nbins=15)
        else:
            plt.plot(
                filtered_df.index, filtered_df[gpu_col],
                linestyle='-', color='b',
                linewidth=1.8, alpha=0.9
            )
            plt.xlabel('Index')

        # --- Y-axis improvements ---
        plt.ylabel(gpu_col)
        plt.ylim(80, 100)  # zoom into 80–100% range
        plt.yticks(range(80, 101, 2))  # tick every 2%
        plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
        plt.minorticks_on()  # show minor grid lines for fine detail

        plt.title('GPU Usage (ollama)')
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)  # high-res output
        print(f"Plot saved to {plot_file}")
    else:
        print("Dataset empty after filtering. No plot generated.")


if __name__ == "__main__":
    process_gpu_data('gemma3_1b.csv', 'gemma3_1b_zero_removed80to100.csv', 'gemma3_1b2.png')
