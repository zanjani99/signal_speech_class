read -p "Enter file path: " FILEPATH
FILES="$FILEPATH/*/*"
if ! "./16kmono_audio/$FILEPATH"; then
  mkdir  -p "./16kmono_audio/$FILEPATH"

fi



for f in $FILES
do
  echo "Processing $f file..."
  # if [ ! -f "$f" ]; then

  ffmpeg -i $f -acodec pcm_s16le -ac 1 -ar 16000 "./16kmono_audio/$f"
  # fi
done
echo -e
# printf "FILES\n"
