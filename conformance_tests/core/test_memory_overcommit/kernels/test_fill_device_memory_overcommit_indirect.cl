struct buffer {
  ulong *data;
};

kernel void fill_device_memory(global struct buffer *in_buffers, ulong alloc_size, ulong value) {
  size_t item_x = get_global_id(0);
  size_t buffer_idx = get_global_id(1);
  size_t num_allocs = get_global_size(1);
  
  size_t per_alloc = alloc_size / sizeof(ulong);
  
  for (int i = 0; i < 16; ++i) {
	size_t idx = item_x * 16 + i;
	if (idx < per_alloc) {
		in_buffers[buffer_idx].data[idx]++;
	}
  }
}