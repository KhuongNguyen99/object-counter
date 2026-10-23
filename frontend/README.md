# Frontend UI workflow

Use this checklist to recreate Countwell's frontend with any layout, colors, or theme. Keep the behavior and workflow below. Use everyday words; users do not need model names or a text prompt.

## 1. Add photos

- Photo input: choose multiple photos or use the device camera. Support JPEG, PNG, and WebP, up to 20 photos per batch, 20 MB per photo, and 30 megapixels per image.
- Photo queue: show previews and allow removal before counting. Explain rejected files while keeping valid photos.
- Examples on each photo: use the uploaded photos themselves for selecting examples. Each photo uses only its own boxes, with no separate reference upload. Switching between photos preserves their selections and marked areas.

## 2. Choose examples

- Manual selection: drag a box around an object to select it. Keep boxes inside the photo and prevent duplicates.
- Mark areas: switch between drawing examples and marking one or more search areas. Allow removing areas and undoing changes.
- Find objects: require marked areas and run only when explicitly requested. Search inside those areas; uploading photos and drawing areas must not trigger detection.
- Filter suggestions: merge duplicate detections from overlapping areas and remove unusually small or large boxes.
- Preselect examples: choose up to 10 suitable boxes spread across the photo, with some randomness. Preserve existing selections and fill toward 10. Use fewer if necessary. Rerunning detection refreshes unselected suggestions.
- Flexible selection: show the selected count and recommend 10, while allowing 1 to 32 distinct examples per photo. Detection results must not inherit the 32-example limit.
- Main photo: show all box outlines, distinguish selected boxes, and highlight the focused box separately. Omit visible box numbers and pending/accepted/removed labels.
- Suggestion browser: show a close-up of the current unselected suggestion, its position, previous/next controls, and select/adjust actions. Selecting removes it from this browser and advances to the next suggestion.
- Selected thumbnails: show only selected boxes, cropped exactly to their boundaries with their proportions preserved. Each has a corner X to deselect. Wrap the list into rows. Clicking a thumbnail focuses its box; deselecting returns it to suggestions.
- Adjust and undo: move or resize a focused box, save or cancel, and undo changes. Done closes the editor; reopening keeps the selections.
- Empty suggestions: let users draw examples manually or change marked areas and retry.

### Image tools

- Photo zoom: scroll to zoom around the pointer. Provide zoom buttons, a percentage, and reset to fit. Middle-mouse dragging pans the photo.
- Pointer magnifier: show a small enlarged view that follows the mouse while hovering or drawing. Keep it within the window, with dotted cross lines extending to its edges.
- Magnifier zoom: Alt + scroll changes only the magnifier's enlargement. Hide it when the main image already shows enough detail, during panning, or outside the image. Hide it for touch input.

### Keyboard controls

Shortcuts work throughout the example editor, including the photo and controls. Keep a small shortcut hint visible and preserve normal typing in input fields.

| Key | Action |
| --- | --- |
| Left / Up arrow | Previous unselected suggestion |
| Right / Down arrow | Next unselected suggestion |
| Space or A | Select the current suggestion and advance |
| X, Delete, or Backspace | Deselect the focused selected box |
| E | Adjust the focused box |
| Arrow keys while adjusting | Move the box |
| Shift + arrow keys while adjusting | Resize the box |
| Enter / Escape while adjusting | Save / cancel the adjustment |
| Ctrl / Cmd + Z | Undo the last selection, box, or marked-area change |
| Escape while marking areas | Leave area-marking mode |

## 3. Count photos

- Start counting: require examples on every queued photo, then count each full photo using its own examples. Search areas limit Find objects, not regular counting. Open the first photo missing examples and explain what is needed.
- Quick count: offer a separate action for simple, separate, evenly spaced objects. Require 1 to 32 marked areas per photo and count suitable detections only inside them, using the same detector as Find objects. Examples and a prior Find objects step are unnecessary. Do not cap results at 10 or 32 objects.
- Match strength: provide a regular-counting slider from 0.01 to 0.95, initially 0.80. Higher values keep closer matches. Save the choice locally across tab closures. It does not control Quick count or Find objects.
- Progress: show waiting, counting, complete, and failed states; wait for backend completion notifications and disable conflicting actions during work.
- Automatic sleep: for remote counting, show a small status in the header and a ten-minute idle countdown after the last job finishes. Check the backend every ten seconds and update the displayed seconds between checks. Counting or finding boxes starts a new idle period when finished; viewing results and status checks do not keep it awake. Show Starting, Counting, Ready, Asleep, or Status unavailable as appropriate. At zero, show Stopping until the backend confirms sleep. Preserve the timer across page and backend restarts; omit the countdown if its start time is unknown.
- Summary: show total objects, the latest batch count or status, photos counted, and number of batches. Include each batch's date and status, and identify Quick count batches.

## 4. Review and correct

- Photo results: draw numbered boxes over the original photo from detection data. Show filename, count, review status, and warnings. Allow preview enlargement and Review photo. A small corner button composes and downloads a full-resolution annotated JPEG only when clicked.
- Corrections: show the original photo with counted boxes. Provide Add box, Remove box, Move photo, zoom, and Undo. Update the displayed count while editing.
- Save and confirm: Save changes stores a draft. Done reviewing saves, marks the photo reviewed, and closes the review. Saved corrections update photo, batch, and overall totals.
- Protect edits: confirm before discarding unsaved changes. If another tab changes the result, preserve local edits, explain the conflict, and offer Reload latest.
- Warnings and retries: flag empty counts and possible misses or overlaps. Keep failed batches visible and excluded from totals; retry by adding those photos again.

## 5. Continue, finish, and recover

- More batches: add photos to the same count through regular counting or Quick count. Each new photo needs its own examples or areas.
- Finish count: wait for running work and require queued photos to be counted or removed. Finalize the total, allow viewing without further edits, and offer New count.
- Download results: export saved photo counts and running totals as CSV during or after a session.
- Restore: reopen the latest saved count after refreshing or reopening the tab. Unsubmitted photos and selections do not survive a reload.
- Errors: retain queued photos after upload failures, retry temporary refresh failures, and explain unavailable saved counts with short, actionable messages.
- Help and accessibility: briefly explain examples, Quick count, file limits, and double counting from overlapping photos. Label icon controls, preserve keyboard focus, and announce selection and status changes.
