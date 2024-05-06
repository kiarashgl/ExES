<script setup lang="ts">
import { reactive, ref, watch } from "vue"
import * as vNG from "v-network-graph"
import {
  ForceLayout,
  ForceNodeDatum,
  ForceEdgeDatum,
} from "v-network-graph/lib/force-layout"
import {VEdgeLabel} from "v-network-graph";
const componentToHex = (c) => {

  const hex = Math.floor(c).toString(16);
  return hex.length == 1 ? "0" + hex : hex;
}

let configs = reactive(
    vNG.defineConfigs({
      view: {
        layoutHandler: new ForceLayout({
          positionFixedByDrag: false,
          // positionFixedByClickWithAltKey: true,
          createSimulation: (d3, nodes, edges) => {
            // d3-force parameters
            console.log("NODES:", nodes)
            console.log("EDGES:", edges)
            const forceLink = d3.forceLink<ForceNodeDatum, ForceEdgeDatum>(edges).id(d => {
              console.log("D:", d)
              return d.id
            })
            return d3
              .forceSimulation(nodes)
              .force("edge", forceLink.distance(40).strength(0.5))
              .force("charge", d3.forceManyBody().strength(-800))
              .force("center", d3.forceCenter().strength(0.05))
              .alphaMin(0.001)
          }
        }),
      },
      edge: {
        label: {
          margin: 4,
          background: {
            visible: true,
            color: "rgba(255,255,255,0.7)",
            padding: {
              vertical: 1,
              horizontal: 4,
            },
            borderRadius: 2,
          },
        },
        normal: {
          width: edge => {
            if ('status' in edge) {
              if (edge.status === "new")
                return 5;
              else if (edge.status === "removed")
                return 5;
              else
                return 1;
            }
            else
              return 2
          },
          color: edge => {
            // console.log("asd")
            if ('color' in edge) {
              const s = "#" + componentToHex(edge.color[0]) + componentToHex(edge.color[1])
                + componentToHex(edge.color[2]) + componentToHex(edge.color[3] * 255)
              console.log(edge, s)
              return s
            }
            else {
              if (edge.status === "new")
                return "#53af00";
              else if (edge.status === "removed")
                return "#e13023";
              else
                return "rgba(199,199,199,0.49)";
            }
          },

          // label: edge => "asd",


          // dasharray: edge => (edge.dashed ? "4" : "0"),
        },
      },
      node: {
        selectable: node => {
          return node.mainNode
        },
        normal: {
          type: "circle",
          radius: node => {
            const mxrank = Math.min(node.rank, 1000)
            const sz = (1000 - mxrank) / 1000 * 7 + 2
            // console.log(node, node.rank)
            return sz
          }, // Use the value of each node object
          color: node => {
            if (node.mainNode)
              return "#378eee"
            else return "#8f8f8f"
          },
        },

        label: {
          // size: 3,
          fontSize: node => node.size,
          visible: true,
        },
      },
    })
  )
</script>

<script lang="ts">
import { VNetworkGraph } from "v-network-graph"
import "v-network-graph/lib/style.css"

export default {
  name: "CollaborationGraph",
  components: {VNetworkGraph},
  props: {
    graph: Object,
    shap: Boolean,
  },
  emits: ['upd'],

  data: function () {
    return {
      sn: undefined
    }
  },
  mounted() {
  }
}
</script>

<style>
.graph {
  width: 100%;
  height: 50vh;
}

</style>

<template>
  <div>
<!--    ammat-->
<!--    {{ sn }}-->
    <v-network-graph
      class="graph"
      :nodes="graph.nodes"
      :edges="graph.edges"
      v-model:selected-nodes="sn"
      @update:selectedNodes="$emit('upd',sn)"
      @nodeclick="alert('abbas')"
      :configs="configs">
      <template #edge-label="{ edge, hovered, ...slotProps }">
        <v-edge-label v-if="edge.shap_value && hovered" :text="edge.shap_value" :class="{hovered}" align="center" vertical-align="center" v-bind="slotProps" />
      </template>
    </v-network-graph>
  </div>
</template>
