using SimpleNN.Graph;

namespace SimpleNN.Module
{
    public class ModuleList : Module
    {
        private Module[] _modules;
        public ModuleList(params Module[] modules)
        {
            int c = modules.Length;
            _modules = new Module[c];
            for (int i = 0; i < c; i++)
            {
                _modules[i] = AddModule(modules[i]);
            }
        }
        public Module this[int i] => _modules[i];
    }
}