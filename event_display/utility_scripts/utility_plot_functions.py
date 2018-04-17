import plotly.plotly as py
import plotly.graph_objs as go
from detector_positions import structured_positions

def plotly_evt_plot(evt, doms_hit, norm_times, mc_positions, **kwargs):
    detfile = "utilities/km3net_jul13_90m.detx"
    doms, pmts = structured_positions(detfile)
    
    def trace(evt, doms_hit, norm_times):
        t = norm_times[evt]
        trace = go.Scatter3d(x = doms_hit[evt]['x'],
                             y = doms_hit[evt]['y'],
                             z = doms_hit[evt]['z'],
                             mode='markers',
                             marker=dict(color=t, colorscale='Bluered', 
                                         size=4, symbol='circle',
                                         line=dict(width=1), opacity=0,
                                         colorbar = go.ColorBar(x=-0.5)))
        return trace

    def mc_start(evt, mc_positions):
        mc_start = go.Scatter3d(x = mc_positions[0][evt]['x'],
                                y = mc_positions[0][evt]['y'],
                                z = mc_positions[0][evt]['z'], 
                                #name = 'mc_start', 
                                marker = dict(size = 3, symbol="star-open-dot", color = 'blue'))
        return mc_start

    def mc_end(evt, mc_positions):
        mc_end = go.Scatter3d(x = mc_positions[1][evt]['x'],
                              y = mc_positions[1][evt]['y'],
                              z = mc_positions[1][evt]['z'],
                              #name = 'mc_end', 
                              marker = dict(size = 3, symbol="star-open-dot", color = 'red'))
        return mc_end

    def mc_track(evt, mc_positions):
        mc_track = go.Scatter3d(x = (mc_positions[0][evt]['x'],mc_positions[1][evt]['x']),
                                y = (mc_positions[0][evt]['y'],mc_positions[1][evt]['y']),
                                z = (mc_positions[0][evt]['z'],mc_positions[1][evt]['z']),
                                mode = 'lines', 
                                #name = 'mc_track', 
                                line = dict(width = 2, color = 'purple'))
    
        return mc_track
    
    #show detector
    trace1 = go.Scatter3d(x= doms['x'],
                          y= doms['y'],
                          z= doms['z'],
        mode='markers', marker=dict(size=6, line=dict(color='rgb(127, 127, 127, 0.14)', width=1),opacity=0.2))
    layout = go.Layout(
                        scene = dict(
                        xaxis = dict(
                             backgroundcolor="rgb(200, 200, 230)",
                             gridcolor="rgb(255, 255, 255)",
                             showbackground=True,
                             zerolinecolor="rgb(255, 255, 255)",),
                        yaxis = dict(
                            backgroundcolor="rgb(230, 200,230)",
                            gridcolor="rgb(255, 255, 255)",
                            showbackground=True,
                            zerolinecolor="rgb(255, 255, 255)"),
                        zaxis = dict(
                            backgroundcolor="rgb(230, 230,200)",
                            gridcolor="rgb(255, 255, 255)",
                            showbackground=True,
                            zerolinecolor="rgb(255, 255, 255)",),),
                        width=700,
                        margin=dict(
                        r=10, l=10,
                        b=10, t=10)
                      )
    data = [trace1, trace(evt, doms_hit, norm_times), 
                mc_start(evt, mc_positions), mc_end(evt, mc_positions), mc_track(evt, mc_positions)]
    fig = go.Figure(data=data, layout= layout)
    return fig
                    
def plotly_detector_plot(**kwargs):
    detfile = "utilities/km3net_jul13_90m.detx"
    doms, pmts = structured_positions(detfile)
    
    #show detector
    detector = go.Scatter3d(x= doms['x'],
                          y= doms['y'],
                          z= doms['z'],
                          mode='markers', marker=dict(size=6, line=dict(color='rgb(127, 127, 127, 0.14)', width=1), 
                                                      opacity=0))

    layout = go.Layout(
                        scene = dict(
                        xaxis = dict(
                             backgroundcolor="rgb(200, 200, 230)",
                             gridcolor="rgb(255, 255, 255)",
                             showbackground=True,
                             zerolinecolor="rgb(255, 255, 255)",),
                        yaxis = dict(
                            backgroundcolor="rgb(230, 200,230)",
                            gridcolor="rgb(255, 255, 255)",
                            showbackground=True,
                            zerolinecolor="rgb(255, 255, 255)"),
                        zaxis = dict(
                            backgroundcolor="rgb(230, 230,200)",
                            gridcolor="rgb(255, 255, 255)",
                            showbackground=True,
                            zerolinecolor="rgb(255, 255, 255)",),),
                        width=700,
                        margin=dict(
                        r=10, l=10,
                        b=10, t=10)
                      )
    fig = go.Figure(data=[detector], layout=layout)
    return fig


def mpl_plot_detector(**kwargs):
    detfile = "utilities/km3net_jul13_90m.detx"
    doms, pmts = structured_positions(detfile)
    
    """
    Function to show all of the doms in the (complete) detector
    The axes limits are set to display the whole detector
    
    Returns:
    --------
    ax : matplotlib Axes3D
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,10))
    ax = Axes3D(fig)
    ax.set_xlim(-400,400)
    ax.set_ylim(-400,400)
    ax.set_zlim(-100,800)
    ax.set_xlabel('X(m)',size=15)
    ax.set_ylabel('Y(m)',size=15)
    ax.set_zlabel('Z(m)',size=15)
    ax.scatter(doms['x'], doms['y'], doms['z'], color="dodgerblue" , alpha=0.50, s=300, edgecolors='w')
    #color='#A2D4FF'
    return ax

def mpl_evt_plot(evt, doms_hit, norm_times, mc_positions,**kwargs):
    detfile = "utilities/km3net_jul13_90m.detx"
    doms, pmts = structured_positions(detfile)
    
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    fig = plt.figure(figsize=(12,8))
    fig.suptitle('Hits on DOMs',size=15)
    ax = Axes3D(fig)
    ax.set_xlim(-400,400)
    ax.set_ylim(-400,400)
    ax.set_zlim(-100,800)
    
    ax.set_xlabel('X(m)',size=15)
    ax.set_ylabel('Y(m)',size=15)
    ax.set_zlabel('Z(m)',size=15)
    ax.scatter(doms['x'], doms['y'], doms['z'], color="dodgerblue" , alpha=0.05, s=300, edgecolors='turquoise')
    
    def scatter(evt, doms_hit, times):#, color=(random.random(), random.random(), random.random())):
      
        """
        Function to plot a numucc event as the hit doms in the evt.
        The color scale from blue to red shows the hit times. Event starts at blue and ends in red

        Parameters:
        -----------
        evt : np.int
             event id to plot
        doms_hit : np.ndarray
            array containing the doms hit positions per event
        times : np.ndarray
            array containing the times of the hits 
        
        Returns:
        --------
        scatter : matplotlib ax.scatter
            scatter plot of the chosen event  
        """
    
        return ax.scatter(doms_hit[evt]['x'],
                          doms_hit[evt]['y'],
                          doms_hit[evt]['z'],
            color=cm.bwr(norm_times[evt]),s=200, alpha=0.9, edgecolor="gray")
    
    def mc_points(evt, mc_positions):
       
        """
        Function to plot the mc event: start point, end point, track
        The color scale from blue to red shows the hit times. Event starts at blue and ends in red

        Parameters:
        -----------
        evt : np.int
             event id to plot
        mc_positions : np.ndarray
            numpy array of the 3D positions of the mc events: mc_positions[0] is the start positions; mc_positions[1] 
            is the ending position of the track
        Returns:
        --------
    
        """
        start = ax.scatter(mc_positions[0][evt]['x'], mc_positions[0][evt]['y'], mc_positions[0][evt]['z'], 
                           color="blue", marker = '*', s=100), 
        end = ax.scatter(mc_positions[1][evt]['x'], mc_positions[1][evt]['y'], mc_positions[1][evt]['z'], 
                         color="red", marker = '>',  s=100)
        #arrows = ax.quiver(mu_start[evt]['x'], mu_start[evt]['y'], mu_start[evt]['z'], 
        #                  mu_end[evt]['x'], mu_end[evt]['y'], mu_end[evt]['z'], length=trks_len[evt])
        lines = ax.plot([mc_positions[0][evt]['x'], mc_positions[1][evt]['x']],
                        [mc_positions[0][evt]['y'], mc_positions[1][evt]['y']], 
                        zs=[mc_positions[0][evt]['z'], mc_positions[1][evt]['z']])
        return
    
    
    def event(evt):
    
        """
        Function to plot a numucc event as the hit doms in the evt.
        The color scale from blue to red shows the hit times. Event starts at blue and ends in red

        Parameters:
        -----------
        evt : np.int
             event id to plot
        Returns:
        --------
        plot : matplotlib scatter
            scatter plot of the chosen numucc event  
        mc : mc_points 
            scatter plot of start and end position according to mc and line to display the mc track
        """
      
        plot = scatter(evt, doms_hit, norm_times)
        mc = mc_points(evt, mc_positions)
        plt.show()
        return plot, mc
    return event(evt)