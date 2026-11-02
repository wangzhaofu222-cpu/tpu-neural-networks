def validate_clip_identity(clips,roles):
    actors=set();seen=set()
    for role,ids in roles.items():
        if actors&set(ids):raise ValueError('Actor split overlap')
        actors.update(ids)
    for c in clips:
        cid=c['clip_id']
        if cid in seen:raise ValueError('Duplicate clip ID')
        seen.add(cid)
        if c['actor_id'] not in roles[c['role']]:raise ValueError('Clip actor does not belong to its split')
        if c.get('audio_clip_id',cid)!=cid or c.get('visual_clip_id',cid)!=cid:raise ValueError('Audio/video clip identity mismatch')
        for f in c.get('frames',[]):
            if f['clip_id']!=cid or f['role']!=c['role'] or f['actor_id']!=c['actor_id']:raise ValueError('Frame crossed clip/actor/split boundary')
    return True
