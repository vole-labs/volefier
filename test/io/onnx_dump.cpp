#include <iostream>
#include "volefier/io/onnx.h"
using namespace volefier; using namespace std;
int main(int argc, char** argv) {
  OnnxGraph g = load_onnx(argv[1]);
  cout << "inputs: "; for (auto&s:g.inputs) cout<<s<<" "; cout<<"\noutputs: "; for(auto&s:g.outputs) cout<<s<<" "; cout<<"\n";
  cout << g.nodes.size() << " nodes:\n";
  for (auto& n : g.nodes) {
    cout << "  " << n.op_type << "  in=["; for(auto&s:n.input)cout<<s<<",";
    cout << "] out=["; for(auto&s:n.output)cout<<s<<","; cout<<"]";
    for(auto&kv:n.attr) cout<<" "<<kv.first<<"="<<kv.second.i<<"/"<<kv.second.f;
    cout << "\n";
  }
  cout << g.initializer.size() << " initializers:\n";
  for (auto& kv : g.initializer) { cout<<"  "<<kv.first<<" dims=["; for(auto d:kv.second.dims)cout<<d<<","; cout<<"] nfloat="<<kv.second.data.size()<<"\n"; }
  return 0;
}
