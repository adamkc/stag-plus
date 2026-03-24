import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from bs4 import BeautifulSoup, Tag

class XMPHandler:
    """
    Handler for XMP sidecar files used to store image metadata

    This class provides methods to read, modify and write XMP files,
    with a focus on managing hierarchical subject tags and ratings.
    """

    @staticmethod
    def is_xmp_file(filename: str) -> bool:
        """Check if a file is an XMP sidecar file based on its extension"""
        _, file_extension = os.path.splitext(filename)
        return file_extension.lower() == ".xmp"

    @staticmethod
    def possible_names_for_image(filename: str) -> List[str]:
        """Generate possible XMP sidecar filenames for an image file"""
        base, _ = os.path.splitext(filename)
        return [
            f"{filename}.xmp",
            f"{base}.xmp"
        ]

    @staticmethod
    def get_xmp_sidecars_for_image(filename: str) -> List[str]:
        """Find existing XMP sidecar files for an image"""
        file_list = []
        for current in XMPHandler.possible_names_for_image(filename):
            if os.path.exists(current):
                file_list.append(current)
        return file_list

    @staticmethod
    def get_xmp_sidecar(filename: str, prefer_short: bool = False) -> Optional[str]:
        """Find a single XMP sidecar file for an image"""
        possible_names = XMPHandler.possible_names_for_image(filename)
        if prefer_short:
            possible_names.reverse()
        for current in possible_names:
            if os.path.exists(current):
                return current
        return None

    @staticmethod
    def create_xmp_sidecar(image_filename: str, prefer_exact_filenames: bool) -> str:
        """Create a new XMP sidecar file for an image"""
        filename, file_extension = os.path.splitext(image_filename)

        if prefer_exact_filenames:
            xmp_name = f"{image_filename}.xmp"
        else:
            xmp_name = f"{filename}.xmp"

        basename = os.path.basename(image_filename)

        xmp_template = """
        <x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
         <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
          <rdf:Description rdf:about=""
            xmlns:exif="http://ns.adobe.com/exif/1.0/"
            xmlns:xmp="http://ns.adobe.com/xap/1.0/"
            xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/">
          </rdf:Description>
         </rdf:RDF>
        </x:xmpmeta>
        """

        soup = BeautifulSoup(xmp_template, "xml")
        desc = soup("rdf:Description")[0]
        desc["xmpMM:DerivedFrom"] = basename

        print(f"Creating XMP sidecar file at {xmp_name}")
        with open(xmp_name, 'w') as f:
            f.write(str(soup))

        return xmp_name


    def __init__(self, xmp_file_path: str):
        """Initialize XMP handler for an existing XMP file"""
        self.path = xmp_file_path

        with open(xmp_file_path, 'r') as f:
            data = f.read()
            self.soup = BeautifulSoup(data, "xml")

        # Ensure required namespaces and structure
        self.ensure_namespace("xmlns:dc", "http://purl.org/dc/elements/1.1/")
        self.subject = self.ensure_keyword_bag("dc:subject")

        self.ensure_namespace("xmlns:lr", "http://ns.adobe.com/lightroom/1.0/")
        self.hierarchical_subject = self.ensure_keyword_bag("lr:hierarchicalSubject")

    def _get_container(self, base_element: Tag) -> Optional[Tag]:
        """Get the rdf:Bag or rdf:Seq container from an element"""
        if base_element("rdf:Bag"):
            return base_element("rdf:Bag")[0]
        elif base_element("rdf:Seq"):
            return base_element("rdf:Seq")[0]
        return None

    def has_subject_prefix(self, prefix: str) -> bool:
        """Check if any subject starts with the given prefix"""
        subjects_container = self._get_container(self.subject)
        if not subjects_container:
            return False
        for item in subjects_container("rdf:li"):
            if item.string and item.string.lower() == prefix.lower():
                return True
        return False

    def ensure_keyword_bag(self, kw_tag: str) -> Tag:
        """Ensure a keyword bag exists, creating it if necessary"""
        desc = self.soup("rdf:Description")[0]
        if len(self.soup(kw_tag)) == 0:
            subj = self.soup.new_tag(kw_tag)
            bag = self.soup.new_tag("rdf:Bag")
            subj.append(bag)
            desc.append(subj)
        return self.soup(kw_tag)[0]

    def ensure_namespace(self, namespace: str, url: str) -> None:
        """Ensure a namespace exists in the XMP file"""
        desc = self.soup("rdf:Description")[0]
        try:
            _ = desc[namespace]
        except KeyError:
            desc[namespace] = url

    def save(self) -> None:
        """Save the XMP file to disk"""
        print(f"Writing to {self.path}")
        if len(str(self.soup)) == 0:
            print("ERROR: Soup creation failed. Not writing XMP file")
            return
        with open(self.path, 'w') as f:
            f.write(str(self.soup))

    def add_single_subject(self, new_subject: str) -> None:
        """Add a single subject tag if it doesn't already exist"""
        subjects_container = self._get_container(self.subject)
        if not subjects_container:
            return
        for item in subjects_container("rdf:li"):
            if item.string == new_subject:
                return
        new_tag = self.soup.new_tag("rdf:li")
        new_tag.string = new_subject
        subjects_container.append(new_tag)

    def remove_subjects_by_prefix(self, prefix: str) -> None:
        """
        Remove all subjects and hierarchical subjects that start with a given prefix.

        Used to clear old iq|/aes| bin tags before writing updated ones,
        so --force doesn't stack duplicate bins.

        Args:
            prefix: The prefix to match (e.g. "iq", "aes")
        """
        # Remove from hierarchical subjects
        hs_container = self._get_container(self.hierarchical_subject)
        if hs_container:
            for item in list(hs_container("rdf:li")):
                if item.string and item.string.startswith(prefix + "|"):
                    item.decompose()

        # Remove from regular subjects
        subjects_container = self._get_container(self.subject)
        if subjects_container:
            # Collect bin names used by iq/aes so we only remove those
            bin_names = {"low", "medium", "high", "poor", "below_average",
                         "average", "good", "excellent"}
            for item in list(subjects_container("rdf:li")):
                if item.string and (item.string == prefix or item.string in bin_names):
                    # Only remove bin-name subjects if they came from this prefix.
                    # The prefix itself (e.g. "iq") is always safe to remove.
                    if item.string == prefix:
                        item.decompose()
                    # For bin names, only remove if the hierarchical version existed
                    # (we already removed those above, so just remove the leaf too)
                    elif item.string in bin_names:
                        item.decompose()

    def add_hierarchical_subject(self, hs: str) -> None:
        """Add a hierarchical subject and its components"""
        hs_container = self._get_container(self.hierarchical_subject)
        if not hs_container:
            return
        for item in hs_container:
            if hasattr(item, 'string') and item.string == hs:
                return
        new_tag = self.soup.new_tag("rdf:li")
        new_tag.string = hs
        hs_container.append(new_tag)
        components = hs.split("|")
        for component in components:
            self.add_single_subject(component)

    def set_rating(self, rating: int) -> None:
        """
        Set the XMP star rating (xmp:Rating).

        This writes to the standard xmp:Rating field which darktable,
        Lightroom, digiKam, and other tools read as star ratings (1-5).

        Args:
            rating: Integer rating from 1 to 5
        """
        rating = max(1, min(5, int(rating)))
        self.ensure_namespace("xmlns:xmp", "http://ns.adobe.com/xap/1.0/")
        desc = self.soup("rdf:Description")[0]
        desc["xmp:Rating"] = str(rating)

    def get_rating(self) -> Optional[int]:
        """
        Get the current XMP star rating.

        Returns:
            Integer rating 1-5, or None if no rating is set
        """
        desc = self.soup("rdf:Description")[0]
        try:
            return int(desc["xmp:Rating"])
        except (KeyError, ValueError):
            return None

    def strip_date_time_original(self) -> None:
        """Remove the DateTimeOriginal tag if it exists"""
        desc = self.soup("rdf:Description")[0]
        try:
            del desc["exif:DateTimeOriginal"]
        except KeyError:
            pass

    def set_output_path(self, new_path: str) -> None:
        """Set a new output path for the XMP file"""
        self.path = new_path

    def get_all_subjects(self) -> List[str]:
        """Get a list of all subjects in the XMP file"""
        subjects = []
        subject_container = self._get_container(self.subject)
        if subject_container:
            for item in subject_container("rdf:li"):
                if item.string:
                    subjects.append(item.string)
        return subjects

if __name__ == '__main__':
    print(XMPHandler.get_xmp_sidecars_for_image("test/P1012424.ORF"))
