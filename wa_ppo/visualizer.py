"""
Visualizer for VRPD Routes
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


class VRPDVisualizer:
    """Visualize VRPD routes for trucks and drones"""
    
    def __init__(self, data_loader, save_dir: str = "./results/"):
        """
        Initialize visualizer
        
        Args:
            data_loader: DataLoader instance with coordinates
            save_dir: Directory to save plots
        """
        self.data_loader = data_loader
        self.save_dir = save_dir
        
        # Get coordinates
        self.x_coords, self.y_coords = data_loader.get_coordinates()
        self.depot_x = self.x_coords[0]
        self.depot_y = self.y_coords[0]
        
        # Colors for multiple trucks
        self.truck_colors = ['blue', 'navy', 'darkblue', 'steelblue', 'cornflowerblue']
        self.drone_colors = ['red', 'darkred', 'crimson', 'indianred', 'lightcoral']
    
    def plot_routes(
        self,
        truck_routes: List[List[int]],  # Now list of lists
        drone_routes: List[List[int]],
        title: str = "VRPD Routes",
        save_path: Optional[str] = None,
    ):
        """
        Plot routes for all vehicles
        
        Args:
            truck_routes: List of routes, one per truck. Each route is list of customer IDs (0-indexed)
            drone_routes: List of routes, one per drone. Each route is list of customer IDs (0-indexed)
            title: Plot title
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(12, 10))
        
        # Plot all customer locations
        plt.scatter(
            self.x_coords[1:], 
            self.y_coords[1:], 
            c='gray', 
            s=100, 
            alpha=0.5, 
            label='Customers',
            zorder=1
        )
        
        # Plot depot
        plt.scatter(
            self.depot_x, 
            self.depot_y, 
            c='green', 
            s=300, 
            marker='*', 
            label='Depot',
            zorder=5,
            edgecolors='black',
            linewidths=2
        )
        
        # Plot each truck route
        for truck_idx, route in enumerate(truck_routes):
            if not route:
                continue
            
            color = self.truck_colors[truck_idx % len(self.truck_colors)]
            
            # Start from depot
            prev_x, prev_y = self.depot_x, self.depot_y
            
            for cid in route:
                # Customer coordinates (cid is 0-indexed, add 1 for coords array)
                curr_x = self.x_coords[cid + 1]
                curr_y = self.y_coords[cid + 1]
                
                # Draw edge
                plt.plot(
                    [prev_x, curr_x], 
                    [prev_y, curr_y], 
                    color=color, 
                    linewidth=2, 
                    alpha=0.7,
                    zorder=2
                )
                
                # Mark customer
                plt.scatter(
                    curr_x, curr_y, 
                    c=color, 
                    s=150, 
                    alpha=0.8,
                    zorder=3,
                    edgecolors='black',
                    linewidths=1
                )
                
                prev_x, prev_y = curr_x, curr_y
            
            # Return to depot
            plt.plot(
                [prev_x, self.depot_x], 
                [prev_y, self.depot_y], 
                color=color, 
                linewidth=2, 
                alpha=0.7,
                linestyle='--',
                zorder=2,
                label=f'Truck-{truck_idx} ({len(route)} customers)'
            )
        
        # Plot each drone route
        for drone_idx, route in enumerate(drone_routes):
            if not route:
                continue
            
            color = self.drone_colors[drone_idx % len(self.drone_colors)]
            
            # Each drone trip: depot -> customer -> depot
            for cid in route:
                curr_x = self.x_coords[cid + 1]
                curr_y = self.y_coords[cid + 1]
                
                # Depot to customer
                plt.plot(
                    [self.depot_x, curr_x], 
                    [self.depot_y, curr_y], 
                    color=color, 
                    linewidth=1.5, 
                    alpha=0.6,
                    linestyle=':',
                    zorder=2
                )
                
                # Mark customer
                plt.scatter(
                    curr_x, curr_y, 
                    c=color, 
                    s=120, 
                    marker='^', 
                    alpha=0.8,
                    zorder=3,
                    edgecolors='black',
                    linewidths=1
                )
                
                # Customer to depot (return)
                plt.plot(
                    [curr_x, self.depot_x], 
                    [self.depot_y, curr_y], 
                    color=color, 
                    linewidth=1.5, 
                    alpha=0.6,
                    linestyle=':',
                    zorder=2
                )
            
            # Add to legend only once per drone
            plt.plot([], [], color=color, linewidth=1.5, linestyle=':', 
                    label=f'Drone-{drone_idx} ({len(route)} customers)')
        
        plt.xlabel('X Coordinate (m)', fontsize=12)
        plt.ylabel('Y Coordinate (m)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.close()
    
    def plot_solution_comparison(
        self,
        solutions: List[dict],
        save_path: Optional[str] = None,
    ):
        """
        Plot multiple solutions side by side for comparison
        
        Args:
            solutions: List of dicts with keys: 'truck_routes', 'drone_routes', 'title'
            save_path: Path to save figure
        """
        n_solutions = len(solutions)
        fig, axes = plt.subplots(1, n_solutions, figsize=(8 * n_solutions, 8))
        
        if n_solutions == 1:
            axes = [axes]
        
        for idx, (ax, sol) in enumerate(zip(axes, solutions)):
            plt.sca(ax)
            
            # Plot customers
            ax.scatter(
                self.x_coords[1:], 
                self.y_coords[1:], 
                c='gray', 
                s=100, 
                alpha=0.5,
                zorder=1
            )
            
            # Plot depot
            ax.scatter(
                self.depot_x, 
                self.depot_y, 
                c='green', 
                s=300, 
                marker='*',
                zorder=5,
                edgecolors='black',
                linewidths=2
            )
            
            # Plot trucks
            for truck_idx, route in enumerate(sol['truck_routes']):
                if not route:
                    continue
                color = self.truck_colors[truck_idx % len(self.truck_colors)]
                prev_x, prev_y = self.depot_x, self.depot_y
                
                for cid in route:
                    curr_x = self.x_coords[cid + 1]
                    curr_y = self.y_coords[cid + 1]
                    ax.plot([prev_x, curr_x], [prev_y, curr_y], 
                           color=color, linewidth=2, alpha=0.7, zorder=2)
                    ax.scatter(curr_x, curr_y, c=color, s=150, alpha=0.8, zorder=3)
                    prev_x, prev_y = curr_x, curr_y
                
                ax.plot([prev_x, self.depot_x], [prev_y, self.depot_y], 
                       color=color, linewidth=2, alpha=0.7, linestyle='--', zorder=2)
            
            # Plot drones
            for drone_idx, route in enumerate(sol['drone_routes']):
                if not route:
                    continue
                color = self.drone_colors[drone_idx % len(self.drone_colors)]
                
                for cid in route:
                    curr_x = self.x_coords[cid + 1]
                    curr_y = self.y_coords[cid + 1]
                    ax.plot([self.depot_x, curr_x], [self.depot_y, curr_y], 
                           color=color, linewidth=1.5, alpha=0.6, linestyle=':', zorder=2)
                    ax.scatter(curr_x, curr_y, c=color, s=120, marker='^', 
                              alpha=0.8, zorder=3)
            
            ax.set_xlabel('X Coordinate (m)', fontsize=10)
            ax.set_ylabel('Y Coordinate (m)', fontsize=10)
            ax.set_title(sol.get('title', f'Solution {idx+1}'), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Comparison plot saved to {save_path}")
        
        plt.close()