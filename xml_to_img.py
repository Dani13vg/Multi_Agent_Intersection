import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import math
import os
import sys # Added for the current_dir logic in the original script

# ... (Keep the parse_sumo_net_xml and generate_intersection_image functions from the previous answer here) ...

# Definition of parse_sumo_net_xml
def parse_sumo_net_xml(xml_file_path):
    """
    Parses a SUMO .net.xml file to extract lane shapes and widths.
    (Full function code from the previous response)
    """
    lanes_data = []
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return None, None, None, None, None
    except FileNotFoundError:
        print(f"Error: File not found {xml_file_path}")
        return None, None, None, None, None

    for lane_element in root.findall(".//lane"):
        shape_str = lane_element.get("shape")
        width_str = lane_element.get("width")

        if not shape_str or not width_str:
            continue

        try:
            width = float(width_str)
        except ValueError:
            print(f"Warning: Lane {lane_element.get('id')} has invalid width '{width_str}'. Skipping.")
            continue

        points_str = shape_str.split(' ')
        shape_coords = []
        for point_pair_str in points_str:
            try:
                x_str, y_str = point_pair_str.split(',')
                x, y = float(x_str), float(y_str)
                shape_coords.append((x, y))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            except ValueError:
                continue
        
        if len(shape_coords) >= 2:
            lanes_data.append({"shape": shape_coords, "width": width})

    if not lanes_data:
        print("No valid lane data found in the XML.")
        location_element = root.find("location")
        if location_element is not None:
            conv_boundary_str = location_element.get("convBoundary")
            if conv_boundary_str:
                try:
                    b_min_x, b_min_y, b_max_x, b_max_y = map(float, conv_boundary_str.split(','))
                    min_x, min_y, max_x, max_y = b_min_x, b_min_y, b_max_x, b_max_y
                except ValueError:
                    print("Warning: Could not parse convBoundary.")
                    return None, None, None, None, None
        if min_x == float('inf'):
             print("Error: Could not determine network boundaries.")
             return None, None, None, None, None
    return lanes_data, min_x, min_y, max_x, max_y

# Definition of generate_intersection_image
def generate_intersection_image(xml_file_path, output_image_path, 
                                pixels_per_meter=5, padding_px=20):
    """
    Generates a binary image of the road intersection from a SUMO .net.xml file.
    (Full function code from the previous response)
    """
    lanes_data, net_min_x, net_min_y, net_max_x, net_max_y = parse_sumo_net_xml(xml_file_path)

    if lanes_data is None:
        print(f"Could not process {xml_file_path}. Image not generated.")
        return

    net_width_meters = net_max_x - net_min_x
    net_height_meters = net_max_y - net_min_y

    img_width_px = math.ceil(net_width_meters * pixels_per_meter) + 2 * padding_px
    img_height_px = math.ceil(net_height_meters * pixels_per_meter) + 2 * padding_px
    
    if img_width_px <= 0 or img_height_px <= 0:
        print(f"Error: Calculated image dimensions are invalid ({img_width_px}x{img_height_px}). Check XML content or parameters.")
        return

    image = Image.new('1', (img_width_px, img_height_px), 0)
    draw = ImageDraw.Draw(image)

    def world_to_pixel(x_world, y_world):
        x_transformed = (x_world - net_min_x) * pixels_per_meter + padding_px
        y_transformed = (net_max_y - y_world) * pixels_per_meter + padding_px # Invert Y
        return int(x_transformed), int(y_transformed)

    for lane in lanes_data:
        shape = lane["shape"]
        lane_width_px = max(1, int(lane["width"] * pixels_per_meter))

        for i in range(len(shape) - 1):
            p1_world = shape[i]
            p2_world = shape[i+1]
            p1_pixel = world_to_pixel(p1_world[0], p1_world[1])
            p2_pixel = world_to_pixel(p2_world[0], p2_world[1])
            draw.line([p1_pixel, p2_pixel], fill=1, width=lane_width_px)
    try:
        image.save(output_image_path)
        print(f"Successfully generated image: {output_image_path}")
    except Exception as e:
        print(f"Error saving image {output_image_path}: {e}")


if __name__ == "__main__":
    # --- Configuration for your specific file ---
    input_xml_file = "sumo/map/x_separate_14m.net.xml"
    
    # Determine the output file name based on the input
    # e.g., "sumo_files/map/simple_inclined_10m.net.xml" -> "simple_inclined_10m_binary.png"
    base_name = os.path.basename(input_xml_file) # "simple_inclined_10m.net.xml"
    name_without_ext = os.path.splitext(base_name)[0] # "simple_inclined_10m.net"
    # If you want to remove .net as well:
    if name_without_ext.endswith(".net"):
        name_without_ext = os.path.splitext(name_without_ext)[0] # "simple_inclined_10m"
        
    output_image_file = f"{name_without_ext}_binary.png"
    
    # You might want to place the output in a specific directory
    # For example, in the same directory as the input XML:
    input_dir = os.path.dirname(input_xml_file)
    if input_dir: # If input_xml_file includes a path
        output_image_path = os.path.join(input_dir, output_image_file)
    else: # If input_xml_file is just a filename (expected in current dir)
        output_image_path = output_image_file

    # Or a dedicated output directory:
    # output_dir = "output_images"
    # os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    # output_image_path = os.path.join(output_dir, output_image_file)

    print(f"Input XML: {input_xml_file}")
    print(f"Output Image: {output_image_path}")

    # --- Image Generation Parameters ---
    resolution_pixels_per_meter = 10  # Adjust for more or less detail
    image_padding_pixels = 50        # Adjust for more or less border

    # --- Run the generation ---
    generate_intersection_image(
        xml_file_path=input_xml_file,
        output_image_path=output_image_path,
        pixels_per_meter=resolution_pixels_per_meter,
        padding_px=image_padding_pixels
    )

    print("Processing complete.")