import shutil
import tempfile

import pytest

from ..dotstyle import *

dot_available = pytest.mark.skipif(
    shutil.which("dot") is None, reason="graphviz 'dot' binary not in PATH"
)


@dot_available
def test_feud():

    # Prepare styles
    def styles():
        resource = DotStyle(shape='rectangle', width=.9, height=.3, fixedsize='true',
                            fontname="Verdana", fontsize=10, penwidth=.5,
                            color=".14 .2 .4", style='filled', fillcolor=".14 0.2 1")
        produce = DotStyle(shape='oval', penwidth=0.5, fontsize=12)

        norm_edge = DotStyle(arrowsize=0.7, penwidth=0.5, color='gray')
        bold_edge = DotStyle(arrowsize=0.6, penwidth=1.5, color='black')
        war_edge = bold_edge + {'color': "0 0.5 1"}
        lord_edge = bold_edge + {'color': 'blue'}

        return locals()

    dotsrc = """
    digraph "Feaudal Processes" {
        rankdir=LR
        ranksep=0.4
        newrank=true
        size = "5.5!"

        graph [label="Complete" labelloc=t]

        edge[norm_edge]
        node[produce]

        {node[resource]
            land peasants goods health protection labor
        }

        subgraph cluster_c1 { label = "" color=white
            Reproduce -> peasants;
            peasants -> Labor;
            {rank = min peasants Labor}
            {rank=same Reproduce }
        }

        goods -> Labor -> {land, labor} -> Work -> goods;

        goods -> Lord -> {protection, land};

        {goods, labor} -> Health -> health;
        {peasants, goods, labor, health} -> Reproduce;

        protection -> War -> {health, goods, land, peasants};


        {rank=min   peasants Labor Lord  }
        {rank=same  land protection labor Reproduce}
        {rank=same  War Work Health  }
    }

    """

    G = IGraph(string=dotsrc, styles=styles())
    with tempfile.SpooledTemporaryFile() as fp:
        G.draw(fp, prog='dot', format='png')


def test_alexnet():
    net_styles = dict(
        conv=DotStyle(shape='invtrapezium', fillcolor='lightblue', style='filled'),
        relu=DotStyle(shape='Msquare', fillcolor='darkolivegreen2', style='filled'),
        conv_relu=DotStyle(color='darkolivegreen2'),
        pool=DotStyle(shape='invtriangle', fillcolor='orange', style='filled'),
        norm=DotStyle(shape='doublecircle', fillcolor='grey', style='filled'),
        full=DotStyle(shape='circle', fillcolor='salmon', style='filled'),
        drop=DotStyle(shape='tripleoctagon', fillcolor='plum2', style='filled'),

        to_conv=DotStyle(color='lightblue', style='bold'),
        to_norm=DotStyle(color='grey', style='bold'),
        to_pool=DotStyle(color='orange', style='bold'),

    )

    dotsrc_net = """
    digraph Alexnet {
        //  GRAPH OPTIONS
        rankdir=TB;         // From Top to Bottom
        size = "15!"

        labelloc="t";       // Tittle possition: top
        label="Alexnet";

        // =======================  NODES  ====================================
        data [shape=box3d, color=black];

        label [shape=tab, color=black];

        loss [shape=component, color=black];

        node [conv];
        conv1;
        conv3;

        node [relu];     // Rectified Linear Unit nodes
        relu1;
        relu3;
        relu6;
        relu7;


        // Splitted layer 2
        // ================
        //
        //  Layers with separated convolutions need to be in subgraphs
        //  This is because we want arrows from individual nodes but
        //  we want to consider all of them as a unique layer.
        //

        subgraph layer2 {
            node [conv];     // Convolution nodes
            conv2_1;
            conv2_2;
            node [relu];     // Rectified Linear Unit nodes
            relu2_1;
            relu2_2;
        }

        // Splitted layer 4
        // ================

        subgraph layer4 {
            node [conv];    // Convolution nodes
            conv3
            relu3
            conv4_1;
            conv4_2;
            node [relu];    // Rectified Linear Unit nodes
            relu4_1;
            relu4_2;
        }

        // Splitted layer 5
        // ================

        subgraph layer5{
            node [conv];    // Convolution nodes
            conv5_1;
            conv5_2;

            node [relu];
            relu5_1;
            relu5_2;
        }


        node [pool];     // Pooling nodes
        pool1;
        pool2;
        pool5;

        node [norm];  // Normalization nodes
        norm1;
        norm2;

        node [full];
        fc6;
        fc7;
        fc8;

        node [drop];
        drop6;
        drop7;

        // ===========================================  LAYESRS =================================

        // LAYER 1
        // -------------------------

        data -> conv1 [to_conv, label="out = 96, kernel = 11, stride = 4"];

        edge [conv_relu];
        conv1 -> relu1;
        relu1 -> conv1;

        conv1 -> norm1 [to_norm, label="local_size = 5, alpha = 0.0001, beta = 0.75"];
        norm1 -> pool1 [to_pool, label="pool = MAX, kernel = 3, stride = 2"];

        pool1 -> conv2_1 [to_conv, label="out = 256, kernel = 5, pad = 2"];
        pool1 -> conv2_2 [to_conv];

        // LAYER 2
        // --------------------------
        edge [conv_relu];
        conv2_1 -> relu2_1;
        conv2_2 -> relu2_2;
        relu2_1 -> conv2_1;
        relu2_2 -> conv2_2;

        conv2_1 -> norm2 [to_norm, label="local_size = 5, alpha = 0.0001, beta = 0.75"];
        conv2_2 -> norm2 [to_norm];
        norm2 -> pool2 [to_pool, label="pool = MAX, kernel = 3, stride = 2"];
        pool2 -> conv3 [to_conv, label="out = 384, kernel = 3, pad = 1"];

        // LAYER 3
        // -------------------------
        conv3 -> relu3 [conv_relu];
        relu3 -> conv3 [conv_relu];
        conv3 -> conv4_1 [to_conv, label="out = 384, kernel = 3, pad = 1"];
        conv3 -> conv4_2 [to_conv];

        // LAYER 4
        // ------------------
        edge [conv_relu];
        conv4_1 -> relu4_1;
        relu4_1 -> conv4_1;
        conv4_2 -> relu4_2;
        relu4_2 -> conv4_2;

        conv4_1 -> conv5_1 [to_conv, label="out = 256, kernel = 3, pad = 1"];
        conv4_2 -> conv5_2 [to_conv];


        // LAYER 5
        // ----------------------
        edge [conv_relu];
        conv5_1 -> relu5_1;
        relu5_1 -> conv5_1;
        conv5_2 -> relu5_2;
        relu5_2 -> conv5_2;

        conv5_1 -> pool5 [to_pool, label="pool = MAX, kernel = 3, stride = 2"];
        conv5_2 -> pool5 [to_pool];

        pool5 -> fc6 [color=salmon, style=bold, label="out = 4096"];
        fc6 -> relu6 [conv_relu];
        relu6 -> fc6 [conv_relu];
        fc6 -> drop6 [color=plum2, style=bold, label="dropout_ratio = 0.5"];
        drop6 -> fc6 [color=plum2];

        // LAYER 6
        // -----------------------
        fc6 -> fc7 [color=salmon, style=bold, label="out = 4096"];

        // LAYER 7
        // ----------------------
        fc7 -> relu7 [conv_relu];
        relu7 -> fc7 [conv_relu];
        fc7 -> drop7 [color=plum2, style=bold, label="dropout_ratio = 0.5"];
        drop7 -> fc7 [color=plum2];
        fc7 -> fc8 [color=salmon, style=bold, label="out = 1000"];

        // LAYER 8
        // ---------------------
        edge [color=black]
        fc8 -> loss;
        label -> loss;
    }
    """

    G = IGraph(string=dotsrc_net, styles=net_styles)
    #G.draw('alex_net.svg', prog='dot')
