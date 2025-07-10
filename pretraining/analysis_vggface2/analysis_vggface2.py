import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict, Counter
import warnings
import time
from tqdm import tqdm 

warnings.filterwarnings('ignore')

# Configuring matplotlib for non-interactive mode (server)
plt.style.use('default')
plt.rcParams['figure.max_open_warning'] = 0

def analyze_dataset_structure(base_path, output_dir):
    
    """ Analyses the basic structure of the dataset. """
    
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the new dataset paths
    datasets = {
        'train': os.path.join(base_path, 'train'),
        'test': os.path.join(base_path, 'test') 
    }
    
    print("="*60)
    print("ANALYSIS OF THE DATASET")
    print("="*60)
    print(f"Results will be stored in: {output_dir}")
    
    # Verify existence of directories
    for split_name, path in datasets.items():
        if not os.path.exists(path):
            print(f"⚠️  WARNING: The {split_name} directory does not exist: {path}")
            return None
    
    return datasets

def count_images_per_class(dataset_path, split_name):
    
    """ Counts images by class in a directory. """
    
    class_counts = defaultdict(int)
    total_images = 0
    corrupted_images = []
    
    if not os.path.exists(dataset_path):
        return class_counts, total_images, corrupted_images
    
    # First, count the total number of image files for tqdm
    estimated_total_images = 0
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            estimated_total_images += sum(1 for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')))

    # Traverse subdirectories (each representing a class/person) with tqdm
    # We use `tqdm(..., total=estimated_total_images, ...)` for the progress bar
    print(f"\nProcessing images in '{split_name}'...")
    
    # Create a list of all image paths to iterate with tqdm
    all_image_paths = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    all_image_paths.append((class_name, os.path.join(class_path, image_file)))

    # Initialising the progress bar
    pbar = tqdm(all_image_paths, total=estimated_total_images, desc=f"Analysing {split_name}", unit="img")

    for class_name, image_path in pbar:
        # Check if the image is corrupted
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify integrity
            class_counts[class_name] += 1
            total_images += 1
        except Exception as e:
            corrupted_images.append({
                'split': split_name,
                'class': class_name,
                'file': os.path.basename(image_path),
                'path': image_path,
                'error': str(e)
            })
        
        # Update progress bar description with progress
        pbar.set_description(f"Analysing {split_name} ({total_images}/{estimated_total_images})")

    pbar.close() # Close progress bar on completion
    
    return dict(class_counts), total_images, corrupted_images

def analyze_image_properties(dataset_path, split_name, sample_size=200):
    
    """ Analyses basic image properties (size, format, etc.). """
    
    image_info = []
    sampled_count = 0
    
    if not os.path.exists(dataset_path):
        return pd.DataFrame()
    
    # Sampling images for analysis
    all_paths_in_split = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    all_paths_in_split.append((class_name, os.path.join(class_path, image_file)))
    
    # Take a random sample
    if len(all_paths_in_split) > sample_size:
        # Use np.random.choice to select random indices
        # This is more efficient than shuffling and then cutting if the list is too large
        random_indices = np.random.choice(len(all_paths_in_split), sample_size, replace=False)
        sampled_image_paths = [all_paths_in_split[i] for i in random_indices]
    else:
        sampled_image_paths = all_paths_in_split

    print(f"Analysing properties of a sample of {len(sampled_image_paths)} images of '{split_name}'...")
    for class_name, image_path in tqdm(sampled_image_paths, desc=f"Sampling {split_name} props", unit="img"):
        try:
            with Image.open(image_path) as img:
                image_info.append({
                    'split': split_name,
                    'class': class_name,
                    'filename': os.path.basename(image_path),
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_bytes': os.path.getsize(image_path)
                })
        except Exception:
            continue
    
    return pd.DataFrame(image_info)

def plot_class_distribution(class_counts_dict, datasets_info, output_dir):
    
    """ Graph the distribution of images by class and save the results.. """
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Changed to 1 row, 3 columns
    fig.suptitle('Distribution of Images by Class', fontsize=22, fontweight='bold')
    
    # Figure 1, 2: Histogram of split distribution (train, test)
    split_names = ['train', 'test'] 
    positions = [0, 1] 
    
    for i, split_name in enumerate(split_names):
        if split_name in class_counts_dict and len(class_counts_dict[split_name]) > 0:
            counts = list(class_counts_dict[split_name].values())
            ax = axes[positions[i]]
            
            ax.hist(counts, bins=20, alpha=0.7, edgecolor='black', color=plt.cm.Set3(i))
            ax.set_title(f'Distribution in: {split_name}', fontsize=22)
            ax.set_xlabel('Number of Images per Class', fontsize=20)
            ax.set_ylabel('Frequency', fontsize=20)
            ax.grid(True, alpha=0.3)
            
            # Set tick label sizes
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            
            # Statistics
            mean_count = np.mean(counts)
            median_count = np.median(counts)
            ax.axvline(mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.1f}')
            ax.axvline(median_count, color='green', linestyle='--', label=f'Median: {median_count:.1f}')
            ax.legend(fontsize=16)
        else:
            # If a split does not exist or is empty, hide the subplot or show a message
            axes[positions[i]].axis('off')
            axes[positions[i]].set_title(f'No hay datos para {split_name}', fontsize=22)
    
    # Figure 3: Comparison of totals by split
    splits = list(datasets_info.keys())
    totals = [datasets_info[split]['total_images'] for split in splits]
    colors = ['skyblue', 'lightcoral']
    
    ax_totals = axes[2]
    ax_totals.bar(splits, totals, color=colors[:len(splits)])
    ax_totals.set_title('Total Images per Split', fontsize=22)
    ax_totals.set_ylabel('Number of Images', fontsize=20)
    
    ax_totals.tick_params(axis='x', labelsize=16)
    ax_totals.tick_params(axis='y', labelsize=16)
    
    # Adding values to the bars
    for i, v in enumerate(totals):
        ax_totals.text(i, v + max(totals)*0.01, str(v), ha='center', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    
    # Save graphic
    plot_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved graphic: {plot_path}")
    
    # Create additional chart: Comparative Boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution boxplot
    split_data = []
    split_labels = []
    
    for split_name in split_names: # 'train', 'test'
        if split_name in class_counts_dict and len(class_counts_dict[split_name]) > 0:
            counts = list(class_counts_dict[split_name].values())
            split_data.append(counts)
            split_labels.append(split_name)
    
    if split_data:
        ax1.boxplot(split_data, labels=split_labels)
        ax1.set_title('Image Distribution by Class (Boxplot)', fontsize=22)
        ax1.set_ylabel('Number of Images per Class', fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
    else:
        ax1.axis('off')
        ax1.set_title('No data for Boxplot', fontsize=22)
    
    # Bar chart: Number of classes per split
    class_counts_per_split = [len(class_counts_dict.get(split, {})) for split in split_names]
    ax2.bar(split_names, class_counts_per_split, color=colors[:len(split_names)])
    ax2.set_title('Number of Classes per Split', fontsize=22)
    ax2.set_ylabel('Number of Classes', fontsize=20)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    
    # Adding values to the bars
    for i, v in enumerate(class_counts_per_split):
        if v > 0:
            ax2.text(i, v + max(class_counts_per_split)*0.01, str(v), ha='center', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    
    # Save second graph
    plot_path2 = os.path.join(output_dir, 'class_distribution_detailed.png')
    plt.savefig(plot_path2, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"   📊 Detailed saved graph: {plot_path2}")
    
    return [plot_path, plot_path2]


def plot_image_properties(image_properties, output_dir):
    
    """ Create graphs of image properties. """
    
    if image_properties.empty:
        return []
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Image Properties', fontsize=22, fontweight='bold')
    
    # Graph 1: Distribution of resolutions
    resolutions = image_properties['width'].astype(str) + 'x' + image_properties['height'].astype(str)
    top_resolutions = resolutions.value_counts().head(10)
    
    axes[0, 0].bar(range(len(top_resolutions)), top_resolutions.values)
    axes[0, 0].set_title('Top 10 Resoluciones', fontsize=22)
    axes[0, 0].set_xlabel('Resolution', fontsize=20)
    axes[0, 0].set_ylabel('Frecuency', fontsize=20)
    axes[0, 0].set_xticks(range(len(top_resolutions)))
    axes[0, 0].set_xticklabels(top_resolutions.index, rotation=45, fontsize=16)
    axes[0, 0].tick_params(axis='y', labelsize=16)
    
    # Figure 2: Distribution of formats
    format_counts = image_properties['format'].value_counts()
    axes[0, 1].pie(format_counts.values, labels=format_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Format Distribution', fontsize=22)
    
    # Figure 3: Distribution of file sizes
    sizes_mb = image_properties['size_bytes'] / (1024 * 1024)
    axes[1, 0].hist(sizes_mb, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('File Size Distribution', fontsize=22)
    axes[1, 0].set_xlabel('Size (MB)', fontsize=20)
    axes[1, 0].set_ylabel('Frecuency', fontsize=20)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', labelsize=16)
    axes[1, 0].tick_params(axis='y', labelsize=16)
    
    # Figure 4: Scatter plot width vs height
    axes[1, 1].scatter(image_properties['width'], image_properties['height'], alpha=0.6)
    axes[1, 1].set_title('Image Dimensions', fontsize=22)
    axes[1, 1].set_xlabel('Width (pixels)', fontsize=20)
    axes[1, 1].set_ylabel('Height (pixels)', fontsize=20)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', labelsize=16)
    axes[1, 1].tick_params(axis='y', labelsize=16)
    
    plt.tight_layout()
    
    # Save graph
    plot_path = os.path.join(output_dir, 'image_properties.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved property graph: {plot_path}")
    
    return [plot_path]


def analyze_class_overlap(class_counts_dict):
    
    print("\n" + "="*50)
    print("CLASS OVERLAP ANALYSIS")
    print("="*50)
    
    train_classes = set(class_counts_dict.get('train', {}).keys())
    test_classes = set(class_counts_dict.get('test', {}).keys()) 
    
    print(f"Classes in train: {len(train_classes)}")
    print(f"Classes in test: {len(test_classes)}")
    
    # Overlaps
    train_test_overlap = train_classes.intersection(test_classes)
    
    print(f"\nOverlap train-test: {len(train_test_overlap)} classes")
    
    # Unique classes
    only_train = train_classes - test_classes 
    only_test = test_classes - train_classes
    
    print(f"\nClasses only in train: {len(only_train)}")
    print(f"Classes only in test: {len(only_test)}")
    
    return {
        'train_classes': train_classes,
        'test_classes': test_classes, 
        'overlaps': {
            'train_test': train_test_overlap 
        }
    }

def generate_summary_report(datasets_info, class_counts_dict, corrupted_images, image_properties):
    
    """ Generates a summary report of the analysis. """
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    # General summary
    total_images = sum(info['total_images'] for info in datasets_info.values())
    total_classes = sum(len(counts) for counts in class_counts_dict.values())
    total_corrupted = len(corrupted_images)
    
    print(f"\n📊 GENERAL STATISTICS:")
    print(f"   Total images: {total_images:,}")
    print(f"   Total number of classes (adding repeated): {total_classes}")
    print(f"   Corrupted images found: {total_corrupted}")
    
    # Split statistics
    print(f"\n📁 STATISTICS BY SPLIT:")
    for split_name, info in datasets_info.items():
        counts = list(class_counts_dict[split_name].values()) if split_name in class_counts_dict else []
        if counts:
            print(f"   {split_name}:")
            print(f"      - Total images: {info['total_images']:,}")
            print(f"      - Total classes: {len(counts)}")
            print(f"      - Average images/class: {np.mean(counts):.1f}")
            print(f"      - Median images/class: {np.median(counts):.1f}")
            print(f"      - Range: {min(counts)} - {max(counts)} images/class")

    # Balance analysis
    print(f"\n⚖️  BALANCE ANALYSIS:")
    for split_name, counts in class_counts_dict.items():
        if counts:
            values = list(counts.values())
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            print(f"   {split_name}: Coefficient of variation = {cv:.2f}")
            if cv < 0.3:
                print(f"      ✅ Dataset well balanced")
            elif cv < 0.7:
                print(f"      ⚠️  Moderately imbalanced dataset")
            else:
                print(f"      ❌ Highly imbalanced dataset")

    # Image properties
    if not image_properties.empty:
        print(f"\n🖼️  IMAGE PROPERTIES (sample):")
        print(f"   Most common resolutions:")
        resolutions = image_properties['width'].astype(str) + 'x' + image_properties['height'].astype(str)
        for res, count in resolutions.value_counts().head(5).items():
            print(f"      - {res}: {count} images")
        
        print(f"   Formats:")
        for fmt, count in image_properties['format'].value_counts().items():
            print(f"      - {fmt}: {count} images")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if total_corrupted > 0:
        print(f"   ⚠️  Remove or repair {total_corrupted} corrupted images")

    # Check balance between splits
    if 'train' in datasets_info and 'test' in datasets_info:
        train_size = datasets_info['train']['total_images']
        test_size = datasets_info['test']['total_images']
        ratio = train_size / test_size if test_size > 0 else 0
        
        if ratio < 2:
            print(f"   ⚠️  Consider increasing the training set (current ratio: {ratio:.1f}:1)")
        elif ratio > 10:
            print(f"   ⚠️  The training set might be too large compared to test")

    return {
        'total_images': total_images,
        'total_classes': total_classes,
        'corrupted_count': total_corrupted,
        'balance_analysis': {split: np.std(list(counts.values())) / np.mean(list(counts.values())) 
                            for split, counts in class_counts_dict.items() if counts}
    }

def main_analysis(base_path, output_dir):
    
    """ Main function running all analysis. """
    
    start_time = time.time()  # Start the timer

    # 1. Verify structure
    datasets = analyze_dataset_structure(base_path, output_dir)
    if datasets is None:
        return

    # 2. Count images per class
    class_counts_dict = {}
    datasets_info = {}
    all_corrupted_images = []

    for split_name, dataset_path in datasets.items():
        # The progress bar is now inside count_images_per_class
        class_counts, total_images, corrupted = count_images_per_class(dataset_path, split_name)
        
        class_counts_dict[split_name] = class_counts
        datasets_info[split_name] = {
            'path': dataset_path,
            'total_images': total_images,
            'total_classes': len(class_counts)
        }
        all_corrupted_images.extend(corrupted)
        
        # The "✅ ... images in ... classes" message is shown after tqdm bar
        print(f"   ✅ '{split_name}': {total_images:,} images in {len(class_counts)} classes")
        if corrupted:
            print(f"   ⚠️  {len(corrupted)} corrupted images found in '{split_name}'")

    # 3. Analyze image properties
    print(f"\n🔍 Analyzing image properties (sample)...")
    all_image_properties = []

    for split_name, dataset_path in datasets.items():
        # The progress bar is now inside analyze_image_properties
        props = analyze_image_properties(dataset_path, split_name, sample_size=200)
        all_image_properties.append(props)

    image_properties = pd.concat(all_image_properties, ignore_index=True)

    # 4. Analyze class overlap
    overlap_analysis = analyze_class_overlap(class_counts_dict)

    # 5. Generate visualizations
    print(f"\n📊 Generating visualizations...")
    plot_paths = plot_class_distribution(class_counts_dict, datasets_info, output_dir)

    # Generate image properties plots
    if not image_properties.empty:
        property_plots = plot_image_properties(image_properties, output_dir)
        plot_paths.extend(property_plots)

    # 6. Save detailed reports
    saved_reports = {}
    general_report_path = os.path.join(output_dir, 'general_report.txt')
    with open(general_report_path, 'w', encoding='utf-8') as f:
        f.write("GENERAL ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total images: {sum(info['total_images'] for info in datasets_info.values())}\n")
        f.write(f"Total classes: {sum(len(counts) for counts in class_counts_dict.values())}\n")
    saved_reports['general'] = general_report_path

    # 7. Generate final report
    summary = generate_summary_report(datasets_info, class_counts_dict, all_corrupted_images, image_properties)

    # 8. Show corrupted images if any
    if all_corrupted_images:
        print(f"\n❌ CORRUPTED IMAGES DETECTED:")
        for img in all_corrupted_images[:10]:  # Show only the first 10
            print(f"   {img['split']}/{img['class']}/{img['file']}: {img['error']}")
        if len(all_corrupted_images) > 10:
            print(f"   ... and {len(all_corrupted_images) - 10} more")

    end_time = time.time()  # Stop the timer
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n✅ Analysis completed. All results saved in: {output_dir}")
    print(f"⏱️ Total analysis time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"📊 Plots generated: {len(plot_paths)}")
    print(f"📄 Reports generated: {len([r for r in saved_reports.values() if r is not None])}")

    return {
        'datasets_info': datasets_info,
        'class_counts': class_counts_dict,
        'corrupted_images': all_corrupted_images,
        'image_properties': image_properties,
        'overlap_analysis': overlap_analysis,
        'summary': summary,
        'saved_files': {
            'plots': plot_paths,
            'reports': saved_reports,
            'output_directory': output_dir
        }
    }


# Example usage
if __name__ == "__main__":
    # Paths
    BASE_PATH = "/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/vggface2"  # Parent directory of train and test
    OUTPUT_DIR = "/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/analysis_vggface2"  # New output directory to avoid overwriting

    # Run full analysis
    results = main_analysis(BASE_PATH, OUTPUT_DIR)
    
    if results:
        print(f"\n✅ Analysis completed successfully.")
        print(f"📁 Check the results at: {OUTPUT_DIR}")
        print(f"\nGenerated files:")
        if results['saved_files']['plots']:
            print(f"  📊 Plots ({len(results['saved_files']['plots'])}):")
            for plot in results['saved_files']['plots']:
                print(f"    - {os.path.basename(plot)}")
        
        if results['saved_files']['reports']:
            print(f"  📄 Reports:")
            for name, path in results['saved_files']['reports'].items():
                if path:
                    print(f"    - {name}: {os.path.basename(path)}")
    else:
        print("❌ The analysis could not be completed. Please check the directory paths.")
