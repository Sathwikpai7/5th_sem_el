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
        # make the figure wider for a longer X-axis
        plt.figure(figsize=(16, 6))

        # sort by time if the column exists
        x_col = time_col if time_col in filtered_df.columns else None
        if x_col:
            filtered_df = filtered_df.sort_values(x_col)

            plt.plot(
                filtered_df[x_col], filtered_df[gpu_col],
                marker='o', linestyle='-', color='b',
                alpha=0.8, markersize=4
            )
            plt.xlabel(x_col)
            plt.xlim(filtered_df[x_col].min() - 1, filtered_df[x_col].max() + 1)
            plt.xticks(rotation=45)
            plt.locator_params(axis='x', nbins=15)  # limit number of ticks
        else:
            plt.plot(
                filtered_df.index, filtered_df[gpu_col],
                marker='o', linestyle='-', color='b',
                alpha=0.8, markersize=4
            )
            plt.xlabel('Index')

        plt.ylabel(gpu_col)
        plt.title('GPU Usage (ollama)')
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
    else:
        print("Dataset empty after filtering. No plot generated.")


if __name__ == "__main__":
    process_gpu_data('gemma3_1b.csv', 'gemma3_1b_zero_removed.csv', 'gemma3_1b.png') 